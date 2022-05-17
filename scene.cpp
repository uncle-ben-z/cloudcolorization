#define BOOST_BIND_NO_PLACEHOLDERS
#include <iostream>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include "pugixml.hpp"
#include "opencv2/opencv.hpp"
#include <Eigen/Core>
#include <Eigen/LU>
#include "happly.h"
#include "cnpy.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


class Scene {
    std::string xml_path;
    pugi::xml_document doc;

    int number_classes;

    double chunk_scale;
    Eigen::Matrix4d chunk_transform;

    Eigen::MatrixXd intrinsics_mat;
    Eigen::MatrixXd origins_mat;
    Eigen::MatrixXd transforms_mat;
    Eigen::MatrixXd directions_mat;

  public:
    std::vector<std::vector<cv::Mat>> images;
    std::vector<Eigen::MatrixXf> depths;
    std::vector<cv::Mat> sharpness;
    std::vector<std::string> labels;

    float pixel_size;
    float focal_length;
    int width;
    int height;
    float scale = 1.0;

    Scene(std::string);
    void parse_agisoft_xml(std::string);
    void write_agisoft_xml(std::string);
    void cache_images(std::vector<std::string>, std::string, std::string, float scale);
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
        compute_uvs(std::vector<double>, std::vector<double>);
    std::vector<double> compute_angles(std::vector<double>);
    std::vector<double> compute_weight(std::vector<double>, std::vector<double>, std::vector<double>,
        std::vector<double>, std::vector<double>);
    void filter_cameras(std::string);
    void colorize_point_cloud(std::string, std::string);
    std::vector<uchar> get_image(int);
    std::string get_label(int);
    void reset(std::string);
};

Scene::Scene(std::string xml_path){
    parse_agisoft_xml(xml_path);
}

void Scene::reset(std::string xml_path){
    intrinsics_mat.resize(0,0);
    origins_mat.resize(0,0);
    transforms_mat.resize(0,0);
    directions_mat.resize(0,0);

    // TODO: this kind of memory freeing necessary?
    for (int i = 0; i < images.size(); i++){
        for(int j = 0; j < images[i].size(); j++)
            images[i][j].release();
        depths[i].resize(0,0);
        sharpness[i].release();
    }
    images.clear();
    depths.clear();
    sharpness.clear();
    labels.clear();

    parse_agisoft_xml(xml_path);
}

void Scene::parse_agisoft_xml(std::string xml_path){
    // parse agisoft xml
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
    std::map<int, Eigen::VectorXd> intrinsics;
    for (pugi::xml_node sensor: doc.child("document").child("chunk").child("sensors").children("sensor")){
        Eigen::VectorXd sensor_values(11);

        if (sensor.child("calibration")) {
            // get and store values
            std::string k1 = sensor.child("calibration").child("k1").text().get();
            std::string k2 = sensor.child("calibration").child("k2").text().get();
            std::string k3 = sensor.child("calibration").child("k3").text().get();
            std::string k4 = sensor.child("calibration").child("k4").text().get();
            std::string p1 = sensor.child("calibration").child("p1").text().get();
            std::string p2 = sensor.child("calibration").child("p2").text().get();
            std::string cx = sensor.child("calibration").child("cx").text().get();
            std::string cy = sensor.child("calibration").child("cy").text().get();

            sensor_values <<
                std::stod(k1.empty() ? "0" : k1),
                std::stod(k2.empty() ? "0" : k2),
                std::stod(k3.empty() ? "0" : k3),
                std::stod(k4.empty() ? "0" : k4),
                std::stod(p1.empty() ? "0" : p1),
                std::stod(p2.empty() ? "0" : p2),
                std::stod(sensor.child("calibration").child("f").text().get()),
                std::stod(sensor.child("calibration").child("resolution").attribute("width").value()),
                std::stod(sensor.child("calibration").child("resolution").attribute("height").value()),
                std::stod(cx.empty() ? "0" : cx),
                std::stod(cy.empty() ? "0" : cy);
        }
        else {
            // invalid sensor configuration without calibration. use placeholder data and handle error later
            sensor_values << 0., 0., 0., 0., 0., 0., -1., -1., -1., 0., 0.;
        }
        int id = std::stoi(sensor.attribute("id").value());
        intrinsics[id] = sensor_values;
    }

    // explicity store some camera parameters
    for (pugi::xml_node property:
            doc.child("document").child("chunk").child("sensors").child("sensor").children("property")){
        if (!std::string("pixel_width").compare(property.attribute("name").value()))
            pixel_size = std::stof(property.attribute("value").value());
        else if (!std::string("focal_length").compare(property.attribute("name").value()))
            focal_length = std::stof(property.attribute("value").value());
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
    for (int i = 0; i < sensor_id.size(); i++) {
        Eigen::VectorXd intrinsic = intrinsics[sensor_id[i]];
        if (intrinsic[6] == -1.) {
            // invalid sensor is used
            std::stringstream error_msg;
            error_msg << "Invalid sensor is used with id " << sensor_id[i] << std::endl;
            throw std::runtime_error(error_msg.str());
        }
        width = intrinsic[7];
        height = intrinsic[8];
        intrinsics_mat.row(i) = intrinsic;
    }
}

void Scene::cache_images
        (std::vector<std::string> imgs_paths, std::string sharpness_path, std::string depths_path, float img_scale){

    scale = img_scale;
    width = int(width * scale);
    height = int(height * scale);
    number_classes = imgs_paths.size();

    int img_count = 0;
    for(std::string label : labels){
        std::cout << "\t\r Caching " << img_count << " of " << labels.size() << std::flush;
        img_count++;

        // cache images
        std::vector<cv::Mat> curr_images;
        for (int i = 0; i < number_classes; i++){
            std::string img_path;

            img_path = imgs_paths[i] + "/" + label + ".JPG";
            if (!std::filesystem::exists(img_path))
                img_path = imgs_paths[i] + "/" + label + ".jpg";
                if (!std::filesystem::exists(img_path))
                    img_path = imgs_paths[i] + "/" + label + ".png";

            cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::stringstream error_msg;
                error_msg << "File not found: " << img_path << std::endl;
                throw std::runtime_error(error_msg.str());
            }
            if (img_scale != 1.0)
                cv::resize(img, img, cv::Size(int(scale * img.cols), int(scale * img.rows)), cv::INTER_LINEAR);
            curr_images.push_back(img);
        }
        images.push_back(curr_images);

        // cache sharpness
        std::string img_path = sharpness_path + "/" + label + ".JPG";
        if (!std::filesystem::exists(img_path))
            img_path = sharpness_path + "/" + label + ".jpg";
        if (!std::filesystem::exists(img_path))
            img_path = sharpness_path + "/" + label + ".png";
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img_scale != 1.0)
            cv::resize(img, img, cv::Size(int(scale * img.cols), int(scale * img.rows)), cv::INTER_LINEAR);
        sharpness.push_back(img);

        // cache depth
        cnpy::NpyArray arr = cnpy::npy_load(depths_path + "/" + label + ".npy");
        float* loaded_data = arr.data<float>();
        size_t nrows = arr.shape[0];
        size_t ncols = arr.shape[1];
        Eigen::MatrixXf dep(nrows, ncols);
        for (int j = 0; j < nrows; j++)
            for (int k = 0; k < ncols; k++)
                dep(j,k) = loaded_data[j*ncols+k];
        depths.push_back(dep);
    }
    std::cout << " caching done." << std::endl;
}


std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> Scene::compute_uvs
        (std::vector<double> point_in, std::vector<double> normal_in){

    // get coordinates and normals
    Eigen::Vector4d point(point_in[0], point_in[1], point_in[2], 1);
    Eigen::Vector3d normal(normal_in[0], normal_in[1], normal_in[2]);

    // apply transformation
    Eigen::VectorXd point_trans = (transforms_mat * point);
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
    Eigen::VectorXd rr6 = rr2.cwiseProduct(rr4);
    Eigen::VectorXd rr8 = rr4.cwiseProduct(rr4);
    Eigen::VectorXd row0 = point_trans_mat.row(0);
    Eigen::VectorXd row1 = point_trans_mat.row(1);

    Eigen::VectorXd px = 1 + intrinsics_mat.array().col(0) * rr2.array() +
                             intrinsics_mat.array().col(1) * rr4.array() +
                             intrinsics_mat.array().col(2) * rr6.array() +
                             intrinsics_mat.array().col(3) * rr8.array();
    px = row0.array() * px.array();
    px = px.array() + intrinsics_mat.array().col(4) * (rr2.array() + 2 * row0.cwiseProduct(row0).array()) +
                               2 * intrinsics_mat.array().col(5) * row0.cwiseProduct(row1).array();

    Eigen::VectorXd py = 1 + intrinsics_mat.array().col(0) * rr2.array() +
                             intrinsics_mat.array().col(1) * rr4.array() +
                             intrinsics_mat.array().col(2) * rr6.array() +
                             intrinsics_mat.array().col(3) * rr8.array();

    py = row1.array() * py.array();
    py = py.array() + intrinsics_mat.array().col(5) * (rr2.array() + 2 * row1.cwiseProduct(row1).array()) +
                               2 * intrinsics_mat.array().col(4) * row0.cwiseProduct(row1).array();

    // get uv coordinates
    Eigen::VectorXd pu = intrinsics_mat.array().col(7) * 0.5 +
                         intrinsics_mat.array().col(9) + px.array() * intrinsics_mat.array().col(6);
                         // TODO: include b1 and b2, affinity and non-orthogonality (skew) coefficients (in pixels),
    Eigen::VectorXd pv = intrinsics_mat.array().col(8) * 0.5 +
                         intrinsics_mat.array().col(10) + py.array() * intrinsics_mat.array().col(6);

    // scale
    pu = pu * scale;
    pv = pv * scale;

    // compute distance to pixel
    Eigen::MatrixXd img_plane_uv = Eigen::MatrixXd::Zero(3, px.size());
    img_plane_uv.row(0) = px;
    img_plane_uv.row(1) = py;
    Eigen::VectorXd distances = (img_plane_uv - point_trans_dist).colwise().norm() * chunk_scale;

    // determine uv mask
    // TODO: maybe use orientation from agisoft xml
    Eigen::VectorXd mask = (0 < pu.array() && pu.array() <  scale * intrinsics_mat(0,7) &&
                               0 < pv.array() && pv.array() < scale * intrinsics_mat(0,8)).cast<double>();

    std::vector<double> pu_out(pu.data(), pu.data() + pu.rows() * pu.cols());
    std::vector<double> pv_out(pv.data(), pv.data() + pv.rows() * pv.cols());
    std::vector<double> distances_out(distances.data(), distances.data() + distances.rows() * distances.cols());
    std::vector<double> mask_out(mask.data(), mask.data() + mask.rows() * mask.cols());

    return {pu_out, pv_out, distances_out, mask_out};
}


std::vector<double> Scene::compute_angles(std::vector<double> normal_in){
    Eigen::Vector3d normal(normal_in[0], normal_in[1], normal_in[2]);

    // compute angle
    Eigen::VectorXd nominator = directions_mat.transpose() * normal;
    Eigen::VectorXd denominator = normal.norm() * directions_mat.colwise().norm();
    Eigen::VectorXd angles = nominator.array() / denominator.array();
    angles = angles.array().acos();
    angles = angles * (180.0/M_PI);

    std::vector<double> angles_out(angles.data(), angles.data() + angles.rows() * angles.cols());

    return angles_out;
}


std::vector<double> Scene::compute_weight(std::vector<double> pu_in, std::vector<double> pv_in,
        std::vector<double> distances_in, std::vector<double> uv_mask_in, std::vector<double> angles_in){

    Eigen::VectorXd pu = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(pu_in.data(), pu_in.size());
    Eigen::VectorXd pv = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(pv_in.data(), pv_in.size());
    Eigen::VectorXd dist = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(distances_in.data(), distances_in.size());
    Eigen::VectorXd angle = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(angles_in.data(), angles_in.size());
    Eigen::VectorXd uv_mask = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(uv_mask_in.data(), uv_mask_in.size());

    // get depth and sharpness
    Eigen::VectorXd visible = Eigen::VectorXd::Zero(pu.size());
    Eigen::VectorXd sharp = Eigen::VectorXd::Zero(pu.size());
    for (int j = 0; j < pu.size(); j++ ){
        if (uv_mask[j] == 0)
            continue;
        sharp(j) = double(int(sharpness[j].at<uchar>(pv[j],pu[j])));
        int img_width = intrinsics_mat(j, 7);
        int img_height = intrinsics_mat(j, 8);
        int row = int(pv[j]*0.25/scale);
        int col = int(pu[j]*0.25/scale);

        // if the image is in landscape format, swap the indices
        if (img_width < img_height)
        {
            row = int(pu[j]*0.25/scale);
            col = int(pv[j]*0.25/scale);
        }
        visible(j) = double(depths[j](row,col)); // 0.25 is the target scale of the depthmaps (due to storage shortness)

    }

    // visibility mask
    double delta = 0.05; // manually tuned 0.05 for mt bridge, 0.03 for others
    visible = visible.array() - dist.array();
    visible = (visible.array().abs() < delta).cast<double>();

    // angle weight
    angle = (angle.array() + 180) * M_PI / 180.0;
    angle = angle.array().cos();
    angle = angle.array().max(angle.array() * 0);

    // resolution weight
    Eigen::VectorXd resolution = dist.array() * pixel_size / focal_length;
    resolution = resolution.minCoeff() / resolution.array();

    // sharpness weight
    sharp = sharp.array() / std::max(sharp.array().maxCoeff(), std::numeric_limits<double>::min());

    // total weight
    Eigen::VectorXd weight = uv_mask.array() * visible.array() * angle.array() * resolution.array() * sharp.array();

    std::vector<double> weight_out(weight.data(), weight.data() + weight.rows() * weight.cols());

    return weight_out;
}


void Scene::filter_cameras(std::string in_cloud){
    // load ply point cloud
    happly::PLYData ply(in_cloud);
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

    // start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    Eigen::VectorXd mask_accu = Eigen::VectorXd::Zero(labels.size());

    omp_set_num_threads(16);
    #pragma omp parallel for
    for(int i = 0; i < xyz.size(); i++){
        // some console output
        if (omp_get_thread_num() == 0 && i % int(xyz.size()/16/100+1) == 0 || i == (xyz.size()/16-1)){
            std::cout << "\t\r" << int(100*i/(xyz.size()/16-1)) << "% of " << xyz.size() << " | Time: " <<
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count()
                << "sec " << std::flush;
        }

        // compute image coordinates
        std::vector<double> point_in{xyz[i][0], xyz[i][1], xyz[i][2]};
        std::vector<double> normal_in{nx[i], ny[i], nz[i]};
        auto [pu_out, pv_out, distances_out, uv_mask_out] = compute_uvs(point_in, normal_in);
        std::vector<double> angles_out = compute_angles(normal_in);

        Eigen::VectorXd uv_mask = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(uv_mask_out.data(), uv_mask_out.size());
        Eigen::VectorXd angle = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(angles_out.data(), angles_out.size());

        // angle weight
        angle = (angle.array() + 180) * M_PI / 180.0;
        angle = angle.array().cos();
        angle = angle.array().max(angle.array() * 0);
        angle = (angle.array().abs() > 0).cast<double>();

        uv_mask = uv_mask.array() * angle.array();

        // accumulate results
        #pragma omp critical
        mask_accu = mask_accu + uv_mask;
    }

    int i = 0;
    // disable unused cameras
    for (pugi::xml_node camera: doc.child("document").child("chunk").child("cameras").children("camera")){
        if (camera.child("transform") && !camera.attribute("enabled")){
            // camera label and sensor (intrinsics)
            printf("%d   %s  %d  \n", i, camera.attribute("label").value(), int(mask_accu[i]));
            i++;

            // continue if camera is used
            if (int(mask_accu[i-1]) > 0)
                continue;
            
            camera.append_attribute("enabled") = "false";
        }
        else if (camera.attribute("enabled"))
            camera.attribute("enabled") = "false";
        else
            camera.append_attribute("enabled") = "false";
    }
}

void Scene::write_agisoft_xml(std::string xml_out_path){
    doc.save_file(xml_out_path.c_str());
}

void Scene::colorize_point_cloud(std::string in_cloud, std::string out_cloud){
    // load ply point cloud
    happly::PLYData ply(in_cloud);
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
        if (omp_get_thread_num() == 0 && i % int(xyz.size()/16/100+1) == 0 || i == (xyz.size()/16-1)){
            std::cout << "\t\r" << int(100*i/(xyz.size()/16-1)) << "% of " << xyz.size() << " | Time: " <<
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count()
                << "sec " << std::flush;
        }

        // compute image coordinates
        std::vector<double> point_in{xyz[i][0], xyz[i][1], xyz[i][2]};
        std::vector<double> normal_in{nx[i], ny[i], nz[i]};
        auto [pu_out, pv_out, distances_out, uv_mask_out] = compute_uvs(point_in, normal_in);

        // convert vectors to eigen
        Eigen::VectorXd pu = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(pu_out.data(), pu_out.size());
        Eigen::VectorXd pv = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(pv_out.data(), pv_out.size());

        // compute angles and weights
        std::vector<double> angles_out = compute_angles(normal_in);
        std::vector<double> weight_out = compute_weight(pu_out, pv_out, distances_out, uv_mask_out, angles_out);
        Eigen::VectorXd weight = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(weight_out.data(), weight_out.size());

        // get defect-wise probabilities from images
        Eigen::MatrixXd probabilities = Eigen::MatrixXd::Zero(pu.size(), number_classes);
        for (int j = 0; j < pu.size(); j++ ){
            if (weight[j] == 0)
                continue;
            Eigen::VectorXd curr_class(number_classes);
            for (int k = 0; k < number_classes; k++)
                curr_class(k) = double(int(images[j][k].at<uchar>(pv[j],pu[j])));
            probabilities.row(j) = curr_class;
        }

        // apply weights
        probabilities = probabilities.array().colwise() * weight.array();

        // accumulate probabilities defect-wise
        Eigen::VectorXd accumulator = probabilities.colwise().sum();
        accumulator.reverseInPlace();

        // get argmax
        Eigen::VectorXd::Index maxIndex;
        double maxNorm = accumulator.array().maxCoeff(&maxIndex);

        // normalize
        accumulator = accumulator.array() / std::max(accumulator.array().sum(), std::numeric_limits<double>::min());

        // defects
        defect[i] = maxIndex;
        crack[i] = accumulator[6];
        spall[i] = accumulator[5];
        corr[i] = accumulator[4];
        effl[i] = accumulator[3];
        vege[i] = accumulator[2];
        cp[i] = accumulator[1];
        back[i] = accumulator[0];

        // confidence
        maxNorm = accumulator[maxIndex];
        accumulator[maxIndex] = 0;
        if (accumulator.array().maxCoeff() == 0)
            conf[i] = 1;
        else
            conf[i] = maxNorm - accumulator.array().maxCoeff();
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

    ply.write(out_cloud, happly::DataFormat::Binary);

    std::cout << std::endl;
}

std::vector<uchar> Scene::get_image(int i){
    auto mat = images[i][0];
    std::vector<uchar> array(mat.rows*mat.cols*mat.channels());
    array.assign(mat.data, mat.data + mat.total()*mat.channels());
    return array;
}

std::string Scene::get_label(int i){
    return labels[i];
}

PYBIND11_MODULE(scene, m) {
    pybind11::class_<Scene>(m, "Scene")
        .def(pybind11::init<const std::string &>())
        .def("parse_agisoft_xml", &Scene::parse_agisoft_xml)
        .def("write_agisoft_xml", &Scene::write_agisoft_xml)
        .def("cache_images", &Scene::cache_images)
        .def("compute_uvs", &Scene::compute_uvs)
        .def("compute_angles", &Scene::compute_angles)
        .def("compute_weight", &Scene::compute_weight)
        .def("filter_cameras", &Scene::filter_cameras)
        .def("colorize_point_cloud", &Scene::colorize_point_cloud)
        .def("get_image", &Scene::get_image)
        .def("reset", &Scene::reset)
        .def_readonly("pixel_size", &Scene::pixel_size)
        .def_readonly("focal_length", &Scene::focal_length)
        .def_readonly("width", &Scene::width)
        .def_readonly("height", &Scene::height)
        .def_readonly("scale", &Scene::scale);
}
