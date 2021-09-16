#define BOOST_BIND_NO_PLACEHOLDERS
#include <iostream>
#include <filesystem>
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
#include "cnpy.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

class CloudColorizer {
    std::string xml_path;

    double chunk_scale;
    Eigen::Matrix4d chunk_transform;

    std::vector<std::string> labels;
    Eigen::MatrixXd intrinsics_mat;
    Eigen::MatrixXd origins_mat;
    Eigen::MatrixXd transforms_mat;

  public:
    Eigen::MatrixXd directions_mat;
    std::vector<std::vector<cv::Mat>> images;
    std::vector<Eigen::MatrixXf> depths;
    std::vector<cv::Mat> sharpness;

    CloudColorizer(std::string);
    void parse_agisoft_xml(std::string);
    void cache_images(std::vector<std::string>, float scale);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
        compute_uv(Eigen::Vector4d, Eigen::Vector3d, float);
};

CloudColorizer::CloudColorizer(std::string xml_path){
    this->parse_agisoft_xml(xml_path);
}

void CloudColorizer::parse_agisoft_xml(std::string xml_path){
    // parse agisoft xml
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xml_path.c_str());
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
    chunk_scale = std::stod(chunk_transform_xml.child("scale").text().get());
    chunk_transform = chunk_transform * chunk_scale;
    chunk_transform(3,3) = 1;
    // chunk translation
    std::string translation = chunk_transform_xml.child("translation").text().get();
    for (int i = 0; i < 3; i++){
        chunk_transform(i,3) = std::stod(translation.substr(0, translation.find(delimiter)));
        translation.erase(0, translation.find(delimiter) + delimiter.length());
    }

    // intrinsics
    std::vector<Eigen::VectorXd> intrinsics;
    for (pugi::xml_node sensor: doc.child("document").child("chunk").child("sensors").children("sensor")){
        Eigen::VectorXd sensor_values(11);
        sensor_values <<
            std::stod(sensor.child("calibration").child("k1").text().get()),
            std::stod(sensor.child("calibration").child("k2").text().get()),
            std::stod(sensor.child("calibration").child("k3").text().get()),
            0, // typically not available
            0, // typically not available
            0, // typically not available
            std::stod(sensor.child("calibration").child("f").text().get()),
            std::stod(sensor.child("calibration").child("resolution").attribute("width").value()),
            std::stod(sensor.child("calibration").child("resolution").attribute("height").value()),
            std::stod(sensor.child("calibration").child("cx").text().get()),
            std::stod(sensor.child("calibration").child("cy").text().get());
        intrinsics.push_back(sensor_values);
    }

    // containers
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> transforms;
    std::vector<Eigen::Vector3d> origins;
    std::vector<Eigen::Vector3d> directions;
    std::vector<int> sensor_id;

    // parse cameras
    for (pugi::xml_node camera: doc.child("document").child("chunk").child("cameras").children("camera")){
        // if transform available
        if (camera.child("transform") && !camera.attribute("enabled")){

            // camera label and sensor (intrinsics)
            labels.push_back(camera.attribute("label").value());
            sensor_id.push_back(std::stoi(camera.attribute("sensor_id").value()));

            // parse transform to matrix
            std::string transform_xml = camera.child("transform").text().get();
            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
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

            //if (transforms.size() == 100)
            //    break;
        }
    }

    // convert to Eigen matrices
    origins_mat = Eigen::MatrixXd::Zero(3,origins.size());
    for (int i = 0; i < origins.size(); i++)
        origins_mat.col(i) = origins[i];

    directions_mat = Eigen::MatrixXd::Zero(3,directions.size());
    for (int i = 0; i < directions.size(); i++)
        directions_mat.col(i) = directions[i];

    transforms_mat = Eigen::MatrixXd::Zero(4*transforms.size(),4);
    for (int i = 0; i < transforms.size(); i++)
        transforms_mat.block(i*4, 0, 4, 4) = transforms[i];

    intrinsics_mat = Eigen::MatrixXd::Zero(sensor_id.size(), 11);
    for (int i = 0; i < sensor_id.size(); i++)
        intrinsics_mat.row(i) = intrinsics[sensor_id[i]];
}

void CloudColorizer::cache_images(std::vector<std::string> imgs_paths, float scale){
    std::cout << "Caching images..." << std::flush;

    for(std::string label : this->labels){
        // cache images
        std::vector<cv::Mat> curr_images;
        for (int i = 0; i < imgs_paths.size(); i++){
            std::string img_path;

            if (imgs_paths[i].find("_mask") != std::string::npos)
                img_path = imgs_paths[i] + "/" + label + ".png";

            // special case: depth map as npy
            else if (imgs_paths[i].find("_depth") != std::string::npos){
                img_path = imgs_paths[i] + "/" + label + ".npy";
                cnpy::NpyArray arr = cnpy::npy_load(img_path);
                float* loaded_data = arr.data<float>();
                size_t nrows = arr.shape[0];
                size_t ncols = arr.shape[1];
                Eigen::MatrixXf dep(nrows, ncols);
                for (int j = 0; j < nrows; j++)
                    for (int k = 0; k < ncols; k++)
                        dep(j,k) = loaded_data[j*ncols+k];
                depths.push_back(dep);
                continue;
            }
            else
                img_path = imgs_paths[i] + "/" + label + ".JPG";
                if (!std::filesystem::exists(img_path))
                    img_path = imgs_paths[i] + "/" + label + ".jpg";

            cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            cv::resize(img, img, cv::Size(int(scale * img.cols), int(scale * img.rows)), cv::INTER_LINEAR);
            if (imgs_paths[i].find("_sharp") != std::string::npos)
                sharpness.push_back(img);
            curr_images.push_back(img);

        }
        images.push_back(curr_images);
    }
    std::cout << " caching done." << std::endl;
}


std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> CloudColorizer::
    compute_uv(Eigen::Vector4d point, Eigen::Vector3d normal, float scale){

    // apply transformation
    Eigen::VectorXd point_trans = (this->transforms_mat * point);
    Eigen::MatrixXd point_trans_tmp = Eigen::Map<Eigen::MatrixXd>(point_trans.data(), 4, point_trans.size() / 4);
    Eigen::MatrixXd point_trans_dist = point_trans_tmp.block(0,0,3,point_trans.size() / 4);
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

    Eigen::VectorXd px = 1 + this->intrinsics_mat.array().col(0) * rr2.array() +
                             this->intrinsics_mat.array().col(1) * rr4.array() +
                             this->intrinsics_mat.array().col(2) * rr6.array() +
                             this->intrinsics_mat.array().col(3) * rr8.array();
    px = row0.array() * px.array();
    px = px.array() + this->intrinsics_mat.array().col(4) * (rr2.array() + 2 * row0.cwiseProduct(row0).array()) +
                               2 * this->intrinsics_mat.array().col(5) * row0.cwiseProduct(row1).array();

    Eigen::VectorXd py = 1 + this->intrinsics_mat.array().col(0) * rr2.array() +
                             this->intrinsics_mat.array().col(1) * rr4.array() +
                             this->intrinsics_mat.array().col(2) * rr6.array() +
                             this->intrinsics_mat.array().col(3) * rr8.array();
    py = row1.array() * py.array();
    py = py.array() + this->intrinsics_mat.array().col(4) * (rr2.array() + 2 * row1.cwiseProduct(row1).array()) +
                               2 * this->intrinsics_mat.array().col(5) * row0.cwiseProduct(row1).array();

    // compute distance to pixel
    Eigen::MatrixXd img_plane_uv = Eigen::MatrixXd::Zero(3, px.size());
    img_plane_uv.row(0) = px;
    img_plane_uv.row(1) = py;
    Eigen::VectorXd distances = (img_plane_uv - point_trans_dist).colwise().norm() * this->chunk_scale;

    // get uv coordinates
    Eigen::VectorXd pu = this->intrinsics_mat.array().col(7) * 0.5 +
                         this->intrinsics_mat.array().col(9) + px.array() * this->intrinsics_mat.array().col(6);
    Eigen::VectorXd pv = this->intrinsics_mat.array().col(8) * 0.5 +
                         this->intrinsics_mat.array().col(10) + py.array() * this->intrinsics_mat.array().col(6);

    // scale
    pu = pu * scale;
    pv = pv * scale;

    // determine uv mask
    Eigen::VectorXd mask = (0 < pu.array() && pu.array() <  scale * this->intrinsics_mat(0,7) &&
                               0 < pv.array() && pv.array() < scale * this->intrinsics_mat(0,8)).cast<double>();

    return {pu, pv, distances, mask};
}


void colorize_point_cloud(const std::vector<std::string> &args){
    // paramters
    std::string xml_path = args[0];
    std::string cloud_path = args[1];

    // image folder paths
    std::vector<std::string> imgs_paths;
    for (int i = 3; i < args.size(); i++)
        imgs_paths.push_back(args[i]);

    float scale = 0.5;
    CloudColorizer cloud_colorizer = CloudColorizer(xml_path);
    cloud_colorizer.cache_images(imgs_paths, scale);

    // load ply point cloud
    happly::PLYData ply(cloud_path);
    std::vector<std::array<double, 3>> xyz = ply.getVertexPositions();

    // get normals
    std::vector<double> nx, ny, nz;
    try{
        nx = ply.getElement("vertex").getProperty<double>("nx");
        ny = ply.getElement("vertex").getProperty<double>("ny");
        nz = ply.getElement("vertex").getProperty<double>("nz");
    } catch (const std::exception& e) {
        nx = ply.getElement("vertex").getProperty<double>("normal_x");
        ny = ply.getElement("vertex").getProperty<double>("normal_y");
        nz = ply.getElement("vertex").getProperty<double>("normal_z");
    }

    // initialize containers
    std::vector<int> defect(xyz.size(),0);
    std::vector<float> crack(xyz.size(),0), spall(xyz.size(),0), corr(xyz.size(),0), effl(xyz.size(),0),
                       vege(xyz.size(),0), cp(xyz.size(),0), back(xyz.size(),0), conf(xyz.size(),0),
                       sharp(xyz.size(),0), dist(xyz.size(),0);

    // start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    omp_set_num_threads(16);
    #pragma omp parallel for
    for(int i = 0; i < xyz.size(); i++){
        // some console output
        if (omp_get_thread_num() == 0 && i % int(xyz.size()/16/100) == 0 || i == (xyz.size()/16-1)){
            std::cout << "\t\r" << int(100*i/(xyz.size()/16-1)) << "% of " << xyz.size() << " | Time: " <<
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count()
                << "sec " << std::flush;
        }

        // get coordinates and normals
        Eigen::Vector4d point(xyz[i][0], xyz[i][1], xyz[i][2], 1);
        Eigen::Vector3d normal(nx[i], ny[i], nz[i]);

        // compute image coordinates
        auto [pu, pv, distances, uv_mask] = cloud_colorizer.compute_uv(point, normal, scale);

        Eigen::VectorXd ones = distances;
        ones.setOnes();

        // compute angle
        Eigen::VectorXd nominator = cloud_colorizer.directions_mat.transpose() * normal;
        Eigen::VectorXd denominator = normal.norm() * cloud_colorizer.directions_mat.colwise().norm();
        Eigen::VectorXd angles = nominator.array() / denominator.array();
        angles = angles.array().acos();
        angles = angles * (180.0/M_PI);

        // angle weight
        Eigen::VectorXd mask_angles = (100 < angles.array() && angles.array() < 260).cast<double>();
        Eigen::VectorXd weight_angles = angles.array() - 180;
        weight_angles = weight_angles * M_PI / 2 / 180.0;
        weight_angles = weight_angles.array().cos();
        //weight_angles = 1 - weight_angles.array().abs();
        //weight_angles = ones;

        /*weight_angles = -Eigen::abs(weight_angles.array());
        weight_angles = Eigen::pow(1.5, weight_angles.array());
        weight_angles = weight_angles.array() * mask_angles.array();
        weight_angles = weight_angles.array() / weight_angles.array().maxCoeff();*/

        // distance weight
        Eigen::VectorXd weight_dist = distances;
        weight_dist = weight_dist.array() * mask_angles.array();
        weight_dist = weight_dist.array() * uv_mask.array();
        weight_dist = ones * weight_dist.array().maxCoeff() - weight_dist;
        weight_dist = weight_dist.cwiseProduct(weight_dist);
        weight_dist = weight_dist.array() * mask_angles.array();
        weight_dist = weight_dist.array() * uv_mask.array();
        weight_dist = weight_dist / std::max(weight_dist.array().maxCoeff(), std::numeric_limits<double>::min());

        // container for accumulated weighted probabilities
        Eigen::VectorXd accumulator = Eigen::VectorXd::Zero(imgs_paths.size()-2);
        Eigen::MatrixXd values = Eigen::MatrixXd::Zero(pu.size(), imgs_paths.size()-2);
        Eigen::VectorXd dep = Eigen::VectorXd::Zero(pu.size());
        Eigen::VectorXd shar = Eigen::VectorXd::Zero(pu.size());

        // apply masks
        pu = uv_mask.array() * pu.array();
        pu = mask_angles.array() * pu.array();
        pv = uv_mask.array() * pv.array();
        pv = mask_angles.array() * pv.array();

        // get values from images
        for (int j = 0; j < pu.size(); j++ ){
            if (weight_dist[j] == 0)
                continue;
            Eigen::VectorXd curr_val(imgs_paths.size()-2);
            for (int k = 0; k < imgs_paths.size()-2; k++)
                curr_val(k) = double(int(cloud_colorizer.images[j][k].at<uchar>(pv[j],pu[j])));///255;
            values.row(j) = curr_val;
        }

        // get depth and sharpness
        for (int j = 0; j < pu.size(); j++ ){
            if (weight_dist[j] == 0)
                continue;
            dep(j) = double(cloud_colorizer.depths[j](int(pv[j]/2),int(pu[j]/2))); // / 2 comes from 2 / 4, 2 for image downsizing and 4 from depthmap downsizing
            shar(j) = double(int(cloud_colorizer.sharpness[j].at<uchar>(pv[j],pu[j])));
        }

        // sharpness weight
        Eigen::VectorXd sharpness = shar;//values.block(0, 7, pu.size(), 1);
        double maxSharpness = sharpness.array().maxCoeff();
        sharpness = sharpness.array() / sharpness.array().maxCoeff();

        // visibility check
        Eigen::VectorXd visible = ((dep.array() - distances.array()).abs() < 0.05).cast<double>();

        // apply weights
        values = values.array().colwise() * visible.array();
        values = values.array().colwise() * weight_angles.array();
        //values = values.array().colwise() * sharpness.array();
        values = values.array().colwise() * weight_dist.array();

        // sum over images
        accumulator = values.colwise().sum();

        // shrinked accumulator
        Eigen::VectorXd accumulator_shrinked(imgs_paths.size()-2); // subtract sharpness and depth
        accumulator_shrinked << accumulator[6], accumulator[5], accumulator[4],
            accumulator[3], accumulator[2], accumulator[1], accumulator[0]; // .reverse()-function was unwilling

        // get argmax
        Eigen::VectorXd::Index maxIndex;
        double maxNorm = accumulator_shrinked.array().maxCoeff(&maxIndex);

        // normalize
        accumulator_shrinked = accumulator_shrinked.array() / std::max(accumulator_shrinked.array().sum(), std::numeric_limits<double>::min());

        // defects
        defect[i] = maxIndex;
        crack[i] = accumulator_shrinked[6];
        spall[i] = accumulator_shrinked[5];
        corr[i] = accumulator_shrinked[4];
        effl[i] = accumulator_shrinked[3];
        vege[i] = accumulator_shrinked[2];
        cp[i] = accumulator_shrinked[1];
        back[i] = accumulator_shrinked[0];

        sharp[i] = maxSharpness;
        dist[i] = distances.array().mean();

        // confidence
        maxNorm = accumulator_shrinked[maxIndex];
        accumulator_shrinked[maxIndex] = 0;
        conf[i] = maxNorm - accumulator_shrinked.array().maxCoeff();
    }

    // export ply
    ply.getElement("vertex").addProperty<int>("defect", defect);
    ply.getElement("vertex").addProperty<float>("confidence", conf);
    ply.getElement("vertex").addProperty<float>("background", back);
    ply.getElement("vertex").addProperty<float>("control_point", cp);
    ply.getElement("vertex").addProperty<float>("vegetation", vege);
    ply.getElement("vertex").addProperty<float>("efflorescence", effl);
    ply.getElement("vertex").addProperty<float>("corrosion", corr);
    ply.getElement("vertex").addProperty<float>("spalling", spall);
    ply.getElement("vertex").addProperty<float>("crack", crack);
    //ply.getElement("vertex").addProperty<float>("max_sharpness", sharp);
    //ply.getElement("vertex").addProperty<float>("min_distances", dist);
    // coverage

    ply.write(args[2], happly::DataFormat::Binary);

    std::cout << std::endl;
}

int main (int argc, char *argv[]){
    // read commandline arguments
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++)
        args.push_back(argv[i]);

    colorize_point_cloud(args);
    return (0);
}


PYBIND11_MODULE(colorize_point_cloud, m) {
    pybind11::class_<CloudColorizer>(m, "CloudColorizer")
        .def(pybind11::init<const std::string &>())
        .def("cache_images", &CloudColorizer::cache_images);

    m.def("colorize_point_cloud", &colorize_point_cloud, R"pbdoc(
        Add main
        Some other explanation about the add function.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}