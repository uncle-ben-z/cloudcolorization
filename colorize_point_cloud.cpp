#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <chrono>
#include <omp.h>
#include "pugixml.hpp"
#include "opencv2/opencv.hpp"
#include <Eigen/Core>
#include <typeinfo>

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
            transforms.push_back(transform);

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

            if (transforms.size() == 1)
                break;
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




    return (0);
}
