# Point Cloud Colorization

### No warranty or support is granted in any way! Use at your own risk. Respect the license. 

This repo hosts C++ code for application-specific cloud colorization. It refers to the backprojection of structural defects automatedly detected in images, i.e. 2D, into a (3D) point cloud. It uses the results from Structure-from-Motion (SfM) as provided by the Agisoft xml-file.

#### Clone
```
git clone git@github.com:uncle-ben-z/cloudcolorization.git --recursive
```

#### Compile
``` linux 
cd cloudcolorization

# in case you forgot "--recursive" above
git submodule init
git submodule update

mkdir build
cd build
cmake ..
make
```
