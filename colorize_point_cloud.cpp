#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <chrono>
#include <omp.h>
#include "pugixml.hpp"
#include "opencv2/opencv.hpp"
#include <Eigen/Core>
#include <typeinfo>
#include <math.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pcl/io/ply_io.h>
#include "happly.h"

int main (int argc, char *argv[])
{
    // paramters
    const char* xml_path = argv[1];
    std::string cloud_path = argv[2];

    // image paths
    std::vector<std::string> imgs_paths;
    for (int i = 4; i < argc; i++)
        imgs_paths.push_back(argv[i]);

    // parse agisoft xml
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xml_path);
    std::string delimiter = " ";

    // chunk transform
    pugi::xml_node chunk_transform_xml = doc.child("document").child("chunk").child("transform");
    Eigen::Matrix4d chunk_transform = Eigen::Matrix4d::Zero();
    // chunk rotation
    std::string transform = chunk_transform_xml.child("rotation").text().get();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++){
            chunk_transform(i,j) = std::stod(transform.substr(0, transform.find(delimiter)));
            transform.erase(0, transform.find(delimiter) + delimiter.length());
        }
    // chunk scale
    double scale = std::stod(chunk_transform_xml.child("scale").text().get());
    chunk_transform = chunk_transform * scale;
    chunk_transform(3,3) = 1;
    // chunk translation
    std::string translation = chunk_transform_xml.child("translation").text().get();
    for (int i = 0; i < 3; i++){
        chunk_transform(i,3) = std::stod(translation.substr(0, translation.find(delimiter)));
        translation.erase(0, translation.find(delimiter) + delimiter.length());
    }

    // containers
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> transforms;
    std::vector<std::vector<cv::Mat>> images;
    std::vector<Eigen::Vector3d> origins;
    std::vector<Eigen::Vector3d> directions;
    std::vector<double> intrinsics;

    std::cout << "Preloading images..." << std::flush;

    // parse cameras
    for (pugi::xml_node camera: doc.child("document").child("chunk").child("cameras").children("camera")){
        // if transform available
        if (camera.child("transform")){
            std::string transform_xml = camera.child("transform").text().get();

            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();

            // parse transform to matrix
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++){
                    transform(i,j) = std::stod(transform_xml.substr(0, transform_xml.find(delimiter)));
                    transform_xml.erase(0, transform_xml.find(delimiter) + delimiter.length());
                }

            transform = transform.inverse().eval();
            transforms.push_back(transform * chunk_transform.inverse().eval());

            // world origin
            auto origin = chunk_transform * transform.inverse().eval();
            Eigen::Vector4d z(0.0,0.0,0.0,1.0);
            Eigen::Vector4d orig = origin * z;
            origins.push_back(orig.block(0,0,3,1));

            // viewing direction
            Eigen::Vector4d ret;
            z(2) = -1;
            ret = orig - origin * z;
            directions.push_back(ret.block(0,0,3,1));
            //std::cout << ret << std::endl;

            // store images
            std::vector<cv::Mat> curr_images;
            for (int i = 0; i < imgs_paths.size(); i++){
                std::string img_path;
                if (imgs_paths[i].find("_mask") != std::string::npos)
                    img_path = imgs_paths[i] + "/" + camera.attribute("label").value() + ".png";
                else
                    img_path = imgs_paths[i] + "/" + camera.attribute("label").value() + ".jpg";

                //std::cout << img_path << std::endl;
                cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
                cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), cv::INTER_LINEAR);
                curr_images.push_back(img);
            }

            images.push_back(curr_images);

            if (transforms.size() == 57)
                break;
        }
    }

    // intrinsics
    pugi::xml_node sensor = doc.child("document").child("chunk").child("sensors").child("sensor");

    intrinsics.push_back(std::stod(sensor.child("calibration").child("k1").text().get()));
    intrinsics.push_back(std::stod(sensor.child("calibration").child("k2").text().get()));
    intrinsics.push_back(std::stod(sensor.child("calibration").child("k3").text().get()));
    intrinsics.push_back(0);
    intrinsics.push_back(0);//std::stod(sensor.child("calibration").child("p1").text().get()));
    intrinsics.push_back(0);//std::stod(sensor.child("calibration").child("p2").text().get()));
    intrinsics.push_back(std::stod(sensor.child("calibration").child("f").text().get()));
    intrinsics.push_back(std::stod(sensor.child("calibration").child("resolution").attribute("width").value()));
    intrinsics.push_back(std::stod(sensor.child("calibration").child("resolution").attribute("height").value()));
    intrinsics.push_back(std::stod(sensor.child("calibration").child("cx").text().get()));
    intrinsics.push_back(std::stod(sensor.child("calibration").child("cy").text().get()));

    // convert to Eigen matrices
    Eigen::MatrixXd origins_mat(3,origins.size());
    for (int i = 0; i < origins.size(); i++)
        origins_mat.col(i) = origins[i];

    Eigen::MatrixXd directions_mat(3,directions.size());
    for (int i = 0; i < directions.size(); i++)
        directions_mat.col(i) = directions[i];

    Eigen::MatrixXd transforms_mat(4*transforms.size(),4);
    for (int i = 0; i < transforms.size(); i++){
        transforms_mat.block(i*4, 0, 4, 4) = transforms[i];
    }

    // load ply point cloud
    happly::PLYData plyIn(cloud_path);
    std::vector<std::array<double, 3>> xyz = plyIn.getVertexPositions();
    std::vector<double> nx = plyIn.getElement("vertex").getProperty<double>("nx");
    std::vector<double> ny = plyIn.getElement("vertex").getProperty<double>("ny");
    std::vector<double> nz = plyIn.getElement("vertex").getProperty<double>("nz");

    // initialize containers
    std::vector<int> defect;
    std::vector<float> crack;
    std::vector<float> spall;
    std::vector<float> corr;
    std::vector<float> effl;
    std::vector<float> vege;
    std::vector<float> cp;
    std::vector<float> back;
    std::vector<float> conf;
    for (int i = 0; i < xyz.size(); i++){
        defect.push_back(0);
        crack.push_back(0);
        spall.push_back(0);
        corr.push_back(0);
        effl.push_back(0);
        vege.push_back(0);
        cp.push_back(0);
        back.push_back(0);
        conf.push_back(0);
    }

    std::cout << " preload done." << std::endl;

    // start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    omp_set_num_threads(16);
    #pragma omp parallel for
    for(int i = 0; i < xyz.size(); i++){
        // some console output
        if (omp_get_thread_num() == 0 && i % int(xyz.size()/16/100) == 0 || i == (xyz.size()/16-1)){
            std::cout << "\t\r" << int(100*i/(xyz.size()/16-1)) << "% of " << xyz.size() << " | Time: " <<
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count() << "sec " << std::flush;
        }

        // get coordinates and normals
        Eigen::Vector4d point(xyz[i][0], xyz[i][1], xyz[i][2], 1);
        Eigen::Vector3d normal(nx[i], ny[i], nz[i]);

        // compute distances to cameras
        Eigen::VectorXd distances = (origins_mat.colwise() - point.head(3)).colwise().norm();

        // apply transformation
        Eigen::VectorXd point_trans = (transforms_mat * point);
        Eigen::MatrixXd point_trans_tmp = Eigen::Map<Eigen::MatrixXd>(point_trans.data(), 4, point_trans.size() / 4);
        point_trans_tmp = point_trans_tmp.array().rowwise() /  point_trans_tmp.row(2).array();
        Eigen::MatrixXd point_trans_mat = point_trans_tmp.block(0,0,2,point_trans.size() / 4);

        // correct for radial distortion
        Eigen::VectorXd rr = point_trans_mat.colwise().norm();
        Eigen::VectorXd ones = rr;
        ones.setOnes();

        Eigen::VectorXd rr2 = rr.cwiseProduct(rr);
        Eigen::VectorXd rr4 = rr2.cwiseProduct(rr2);
        Eigen::VectorXd rr6 = rr2.cwiseProduct(rr2).cwiseProduct(rr2);
        Eigen::VectorXd rr8 = rr4.cwiseProduct(rr4);
        Eigen::VectorXd row0 = point_trans_mat.row(0);
        Eigen::VectorXd row1 = point_trans_mat.row(1);

        Eigen::VectorXd px = ones + intrinsics[0] * rr2 +
                                    intrinsics[1] * rr4 +
                                    intrinsics[2] * rr6 +
                                    intrinsics[3] * rr8;
        px = row0.array() * px.array();
        px = px + intrinsics[4] * (rr2 + 2 * row0.cwiseProduct(row0)) +
                                   2 * intrinsics[5] * row0.cwiseProduct(row1);

        Eigen::VectorXd py = ones + intrinsics[0] * rr2 +
                                    intrinsics[1] * rr4 +
                                    intrinsics[2] * rr6 +
                                    intrinsics[3] * rr8;
        py = row1.array() * py.array();
        py = py + intrinsics[4] * (rr2 + 2 * row1.cwiseProduct(row1)) +
                                   2 * intrinsics[5] * row0.cwiseProduct(row1);

        // get uv coordinates
        Eigen::VectorXd pu = ones * intrinsics[7] * 0.5 + ones * intrinsics[9] + px * intrinsics[6];
        Eigen::VectorXd pv = ones * intrinsics[8] * 0.5 + ones * intrinsics[10] + py * intrinsics[6];

        // scale
        pu = pu * 0.5;
        pv = pv * 0.5;

        // determine uv mask
        Eigen::VectorXd mask_uv = (0 < pu.array() && pu.array() < 0.5 * intrinsics[7] &&
                                   0 < pv.array() && pv.array() < 0.5 * intrinsics[8]).cast<double>();

        // compute angle
        Eigen::VectorXd nominator = directions_mat.transpose() * normal;
        Eigen::VectorXd denominator = normal.norm() * directions_mat.colwise().norm();
        Eigen::VectorXd angles = nominator.array() / denominator.array();
        angles = angles.array().acos();
        angles = angles * (180.0/M_PI);

        // determine angle mask
        Eigen::VectorXd mask_angles = (100 < angles.array() && angles.array() < 260).cast<double>();

        // apply masks
        pu = mask_uv.array() * pu.array();
        pu = mask_angles.array() * pu.array();
        pv = mask_uv.array() * pv.array();
        pv = mask_angles.array() * pv.array();

        // compute distance weight
        Eigen::VectorXd weight = distances;
        weight = weight.array() * mask_angles.array();
        weight = weight.array() * mask_uv.array();
        weight = ones * weight.array().maxCoeff() - weight;
        weight = weight.array() * mask_angles.array();
        weight = weight.array() * mask_uv.array();
        weight = weight.cwiseProduct(weight);
        weight = weight / weight.sum();

        // container for accumulated weighted probabilities
        Eigen::VectorXd accumulator = Eigen::VectorXd::Zero(imgs_paths.size());
        Eigen::MatrixXd values = Eigen::MatrixXd::Zero(pu.size(), imgs_paths.size());

        // get values from images
        for (int j = 0; j < pu.size(); j++ ){
            if (weight[j] == 0)
                continue;
            Eigen::VectorXd curr_val(imgs_paths.size());

            for (int k = 0; k < imgs_paths.size(); k++){
                curr_val(k) = double(int(images[j][k].at<uchar>(pv[j],pu[j])));///255;
            }
            values.row(j) = curr_val;
        }

        // sharpness weight
        Eigen::VectorXd sharpness = values.block(0, 7, pu.size(), 1);
        sharpness = sharpness / sharpness.sum();

        // apply weights
        values = values.array().colwise() * sharpness.array();
        values = values.array().colwise() * weight.array();

        // sum over images
        accumulator = values.colwise().sum();

        // zero out image quality (mask) value
        accumulator[7] = 0;

        // get argmax
        Eigen::VectorXd::Index   maxIndex;
        double maxNorm = accumulator.array().maxCoeff(&maxIndex);

        // defects
        defect[i] = 6-maxIndex;
        crack[i] = accumulator[6-6];
        spall[i] = accumulator[6-5];
        corr[i] = accumulator[6-4];
        effl[i] = accumulator[6-3];
        vege[i] = accumulator[6-2];
        cp[i] = accumulator[6-1];
        back[i] = accumulator[6-0];

        // confidence
        accumulator[maxIndex] = 0;
        conf[i] = maxNorm - accumulator.array().maxCoeff();
    }

    // export ply
    plyIn.getElement("vertex").addProperty<int>("defect", defect);
    plyIn.getElement("vertex").addProperty<float>("confidence", conf);
    plyIn.getElement("vertex").addProperty<float>("background", back);
    plyIn.getElement("vertex").addProperty<float>("control_point", cp);
    plyIn.getElement("vertex").addProperty<float>("vegetation", vege);
    plyIn.getElement("vertex").addProperty<float>("efflorescence", effl);
    plyIn.getElement("vertex").addProperty<float>("corrosion", corr);
    plyIn.getElement("vertex").addProperty<float>("spalling", spall);
    plyIn.getElement("vertex").addProperty<float>("crack", crack);

    plyIn.write(argv[3], happly::DataFormat::Binary);

    std::cout << std::endl;

    return (0);
}
