# Point Cloud Colorization

### No warranty or support is granted in any way! Use at your own risk. Respect the license. 

This repo hosts C++ code for application-specific cloud colorization. It refers to the backprojection of structural defects automatedly detected in images, i.e. 2D, into a (3D) point cloud. It uses the results from Structure-from-Motion (SfM) as provided by the Agisoft xml-file.

#### Clone
```
git clone git@github.com:uncle-ben-z/cloudcolorization-fast.git --recursive
```

#### Compile
``` linux 
cd cloudcolorization-fast

# in case you forgot "--recursive" above
git submodule init
git submodule update

mkdir build
cd build
cmake ..
make
```

#### Run
```
./colorize_point_cloud \
[path_to_agisoft_xml] \
[path_to_ply_point_cloud] \
[out_path_for_resulting_point_cloud] \
[paths_to_image_folders_of_defect_class] \
...
[path_to_folder_with_sharpness_images] \
[path_to_folder_with_depth_images]  
```
