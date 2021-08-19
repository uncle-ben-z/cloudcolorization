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

int main ()
{
    // parse agisoft xml
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/cameras.xml");
    std::string delimiter = " ";

    // chunk transform
    pugi::xml_node chunk_transform_xml = doc.child("document").child("chunk").child("transform");
    Eigen::Matrix4f chunk_transform = Eigen::Matrix4f::Zero();
    // chunk rotation
    std::string transform = chunk_transform_xml.child("rotation").text().get();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++){
            chunk_transform(i,j) = std::stof(transform.substr(0, transform.find(delimiter)));
            transform.erase(0, transform.find(delimiter) + delimiter.length());
        }
    // chunk scale
    float scale = std::stof(chunk_transform_xml.child("scale").text().get());
    chunk_transform = chunk_transform * scale;
    chunk_transform(3,3) = 1;
    // chunk translation
    std::string translation = chunk_transform_xml.child("translation").text().get();
    for (int i = 0; i < 3; i++){
        chunk_transform(i,3) = std::stof(translation.substr(0, translation.find(delimiter)));
        translation.erase(0, translation.find(delimiter) + delimiter.length());
    }

    // image paths
    std::vector<std::string> imgs_paths;
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/1_crack");
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/2_spall");
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/3_corr");
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/4_effl");
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/5_vege");
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/6_cp");
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/8_background");
    imgs_paths.push_back("/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/9_mask");

    // containers
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transforms;
    std::vector<std::vector<cv::Mat>> images;
    std::vector<Eigen::Vector3f> origins;
    std::vector<Eigen::Vector3f> directions;
    std::vector<float> intrinsics;


    // parse cameras
    for (pugi::xml_node camera: doc.child("document").child("chunk").child("cameras").children("camera")){
        // if transform available
        if (camera.child("transform")){
            std::string transform_xml = camera.child("transform").text().get();

            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

            // parse transform to matrix
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++){
                    transform(i,j) = std::stof(transform_xml.substr(0, transform_xml.find(delimiter)));
                    transform_xml.erase(0, transform_xml.find(delimiter) + delimiter.length());
                }

            transform = transform.inverse().eval();
            transforms.push_back(transform * chunk_transform.inverse().eval());

            // world origin
            auto origin = chunk_transform * transform.inverse().eval();
            Eigen::Vector4f z(0.0,0.0,0.0,1.0);
            Eigen::Vector4f orig = origin * z;
            origins.push_back(orig.block(0,0,3,1));

            // viewing direction
            Eigen::Vector4f ret;
            z(2) = -1;
            ret = orig - origin * z;
            directions.push_back(ret.block(0,0,3,1));
            std::cout << ret << std::endl;

            // store images
            std::vector<cv::Mat> curr_images;
            for (int i = 0; i < imgs_paths.size(); i++){
                std::string img_path;
                if (imgs_paths[i].find("_mask") != std::string::npos)
                    img_path = imgs_paths[i] + "/" + camera.attribute("label").value() + ".png";
                else
                    img_path = imgs_paths[i] + "/" + camera.attribute("label").value() + ".jpg";

                std::cout << img_path << std::endl;
                cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
                cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), cv::INTER_NEAREST);
                curr_images.push_back(img);
            }

            images.push_back(curr_images);

            //if (transforms.size() == 12)
            //    break;
        }
    }

    // intrinsics
    pugi::xml_node sensor = doc.child("document").child("chunk").child("sensors").child("sensor");

    intrinsics.push_back(std::stof(sensor.child("calibration").child("k1").text().get()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("k2").text().get()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("k3").text().get()));
    intrinsics.push_back(0);
    intrinsics.push_back(std::stof(sensor.child("calibration").child("p1").text().get()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("p2").text().get()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("f").text().get()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("resolution").attribute("width").value()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("resolution").attribute("height").value()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("cx").text().get()));
    intrinsics.push_back(std::stof(sensor.child("calibration").child("cy").text().get()));

    /*for (int i = 0; i < intrinsics.size(); i++)
        std::cout << i << "  " << intrinsics[i] << std::endl;*/


    Eigen::MatrixXf origins_mat(3,origins.size());
    for (int i = 0; i < origins.size(); i++)
        origins_mat.col(i) = origins[i];
    std::cout << "origins: " << origins_mat << std::endl;

    Eigen::MatrixXf directions_mat(3,directions.size());
    for (int i = 0; i < directions.size(); i++)
        directions_mat.col(i) = directions[i];
    std::cout << "directions: " << directions_mat << std::endl;

    Eigen::MatrixXf transforms_mat(4*transforms.size(),4);
    for (int i = 0; i < transforms.size(); i++){
        transforms_mat.block(i*4, 0, 4, 4) = transforms[i];
    }


    // load ply point cloud
    happly::PLYData plyIn("/home/chrisbe/Desktop/rebars_small_medium.ply");
    std::vector<std::array<double, 3>> xyz = plyIn.getVertexPositions();
    std::vector<float> nx = plyIn.getElement("vertex").getProperty<float>("nx");
    std::vector<float> ny = plyIn.getElement("vertex").getProperty<float>("ny");
    std::vector<float> nz = plyIn.getElement("vertex").getProperty<float>("nz");

    // container
    std::vector<int> defect;

    //omp_set_num_threads(4);
    //#pragma omp parallel for
    for(int i = 0; i < xyz.size(); i++){
        std::cout << i << " of " << xyz.size() << std::endl;
        //const int id = omp_get_thread_num();

        // get coordinates and normals
        Eigen::Vector4f point(xyz[i][0], xyz[i][1], xyz[i][2], 1);
        Eigen::Vector3f normal(nx[i], ny[i], nz[i]);

        // compute distances to cameras
        auto distances = (origins_mat.colwise() - point.head(3)).colwise().norm();

        // apply transformation
        Eigen::VectorXf point_trans = (transforms_mat * point);
        Eigen::MatrixXf point_trans_mat = Eigen::Map<Eigen::MatrixXf>(point_trans.data(), 4, point_trans.size() / 4);
        point_trans_mat = point_trans_mat.array().rowwise() /  point_trans_mat.row(2).array();
        point_trans_mat = point_trans_mat.block(0,0,2,point_trans.size() / 4);

        // correct for radial distortion
        Eigen::VectorXf rr = point_trans_mat.colwise().norm();
        auto ones = rr;
        ones.setOnes();

        Eigen::VectorXf rr2 = rr.cwiseProduct(rr);
        Eigen::VectorXf rr4 = rr2.cwiseProduct(rr2);
        Eigen::VectorXf rr6 = rr2.cwiseProduct(rr2).cwiseProduct(rr2);
        Eigen::VectorXf rr8 = rr4.cwiseProduct(rr4);
        Eigen::VectorXf row0 = point_trans_mat.row(0).array();
        Eigen::VectorXf row1 = point_trans_mat.row(1);

        Eigen::VectorXf px = ones + intrinsics[0] * rr2 +
                                    intrinsics[1] * rr4 +
                                    intrinsics[2] * rr6 +
                                    intrinsics[3] * rr8;
        px = row0.array() * px.array();
        px = px + intrinsics[4] * (rr2 + 2 * row0.cwiseProduct(row0)) +
                                   2 * intrinsics[5] * row0.cwiseProduct(row1);

        Eigen::VectorXf py = ones + intrinsics[0] * rr2 +
                                    intrinsics[1] * rr4 +
                                    intrinsics[2] * rr6 +
                                    intrinsics[3] * rr8;
        py = row1.array() * py.array();
        py = py + intrinsics[4] * (rr2 + 2 * row1.cwiseProduct(row1)) +
                                   2 * intrinsics[5] * row0.cwiseProduct(row1);

        // get uv coordinates
        Eigen::VectorXf pu = ones * intrinsics[7] * 0.5 + ones * intrinsics[9] + px * intrinsics[6];
        Eigen::VectorXf pv = ones * intrinsics[8] * 0.5 + ones * intrinsics[10] + py * intrinsics[6];

        pu = pu * 0.5;
        pv = pv * 0.5;

        // determine uv mask
        Eigen::VectorXf mask_uv = (0 < pu.array() && pu.array() < 0.5 * intrinsics[7] &&
                                   0 < pv.array() && pv.array() < 0.5 * intrinsics[8]).cast<float>();


        // compute angle
        Eigen::VectorXf nominator = directions_mat.transpose() * normal;
        Eigen::VectorXf denominator = normal.norm() * directions_mat.colwise().norm();
        Eigen::VectorXf angles = nominator.array() / denominator.array();
        angles = angles.array().acos();
        angles = angles * (180.0/M_PI);

        // determine angle mask
        Eigen::VectorXf mask_angles = (100 < angles.array() && angles.array() < 260).cast<float>();

        // apply mask
        pu = mask_uv.array() * pu.array();
        pv = mask_uv.array() * pv.array();

        // compute weight
        Eigen::VectorXf weight = distances;
        weight = weight.array() * mask_angles.array();
        weight = weight.array() * mask_uv.array();
        weight = ones * weight.array().maxCoeff() - weight;
        weight = weight.array() * mask_angles.array();
        weight = weight.array() * mask_uv.array();
        weight = weight.array() * weight.array();
        weight = weight.array() * weight.array();
        weight = weight.array() * weight.array();
        weight = weight / weight.sum();


        // container for accumulated weighted probabilities
        Eigen::VectorXf accumulator = Eigen::VectorXf::Zero(imgs_paths.size());

        for (int j = 0; j < pu.size(); j++ ){
            if (weight[j] == 0)
                continue;
            Eigen::VectorXf curr_val(imgs_paths.size());

            for (int k = 0; k < imgs_paths.size(); k++){
                curr_val(k) = float(int(images[j][k].at<uchar>(pv[j],pu[j])));///255;
            }

            // apply weight and accumulate
            curr_val = weight[j] * curr_val;
            accumulator = accumulator + curr_val;
        }

        // zero out image quality (mask) value
        accumulator[7] = 0;

        // get argmax
        Eigen::VectorXf::Index   maxIndex;
        float maxNorm = accumulator.array().maxCoeff(&maxIndex);

        defect.push_back(maxIndex);
    }


    // export ply
    plyIn.getElement("vertex").addProperty<int>("defect", defect);
    plyIn.write("/home/chrisbe/Desktop/rebars_small_medium_out.ply", happly::DataFormat::Binary);


    return (0);
}
