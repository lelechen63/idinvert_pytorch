#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

void calcIndexMap( const Eigen::Array2Xf &coords, const Eigen::Array3Xi &coordIdxs, const cv::Size &texSize,
                   cv::Mat &indexMap )
{
    // IndexMap: 3 channels for each pixel, 1: triId, 2 and 3: baycentric coordinates
    indexMap.create( texSize, CV_32FC3 );
    indexMap.setTo( cv::Vec3f( -1.f, -1.f, -1.f ) );

    // For each triangle, find the coverred pixels
    int triNum = int( coordIdxs.cols() );
    for( int triId=0; triId<triNum; triId++ ){
        const Eigen::Vector3i& coordIdx = coordIdxs.col( triId );

        Eigen::Vector2i bboxMin( texSize.width, texSize.height );
        Eigen::Vector2i bboxMax( 0, 0 );

        for( int i=0; i<3; i++ ){
            const Eigen::Vector2f& coord = coords.col( coordIdx[ i ] );
            int x = int( coord[ 0 ] * texSize.width );
            int y = int( coord[ 1 ] * texSize.height );
            bboxMin[ 0 ] = std::min( bboxMin[ 0 ], x );
            bboxMin[ 1 ] = std::min( bboxMin[ 1 ], y );
            bboxMax[ 0 ] = std::max( bboxMax[ 0 ], x );
            bboxMax[ 1 ] = std::max( bboxMax[ 1 ], y );
        }

        bboxMin[ 0 ] = std::min( texSize.width - 1, std::max( 0, bboxMin[ 0 ] ) );
        bboxMin[ 1 ] = std::min( texSize.height - 1, std::max( 0, bboxMin[ 1 ] ) );
        bboxMax[ 0 ] = std::min( texSize.width - 1, std::max( 0, bboxMax[ 0 ] ) );
        bboxMax[ 1 ] = std::min( texSize.height - 1, std::max( 0, bboxMax[ 1 ] ) );

        Eigen::Vector2f vec01 = ( Eigen::Vector2f )coords.col( coordIdx[ 1 ] ) - ( Eigen::Vector2f )coords.col( coordIdx[ 0 ] );
        Eigen::Vector2f vec02 = ( Eigen::Vector2f )coords.col( coordIdx[ 2 ] ) - ( Eigen::Vector2f )coords.col( coordIdx[ 0 ] );

        for( int v = bboxMin[ 1 ]; v <= bboxMax[ 1 ]; v++ ){
            for( int u = bboxMin[ 0 ]; u <= bboxMax[ 0 ]; u++ ){
                Eigen::Vector2f uvCoord( ( u + 0.5f ) / float( texSize.width ), ( v + 0.5f ) / float( texSize.height ) );
                Eigen::Vector2f pt = uvCoord - ( Eigen::Vector2f )coords.col( coordIdx[ 0 ] );

                float denom = vec01[ 0 ] * vec02[ 1 ] - vec01[ 1 ] * vec02[ 0 ];
                float lambda_2 = (pt[ 0 ] * vec02[ 1 ] - pt[ 1 ] * vec02[ 0 ]) / denom;
                float lambda_3 = (vec01[ 0 ] * pt[ 1 ] - vec01[ 1 ] * pt[ 0 ]) / denom;
                float lambda_1 = 1.f - lambda_2 - lambda_3;

                if( lambda_1 > 0.f && lambda_1 < 1.f &&
                    lambda_2 > 0.f && lambda_2 < 1.f &&
                    lambda_3 > 0.f && lambda_3 < 1.f ){
                    indexMap.at<cv::Vec3f>( v, u ) = cv::Vec3f( triId + 0.5f, lambda_1, lambda_2 );
                }
            }
        }
    }
}

py::array_t<float> calcIndexMap_py( const Eigen::Array2Xf& coords, const Eigen::Array3Xi& coordIdxs, int texWidth, int texHeight )
{
    cv::Mat idxMapMat;
    calcIndexMap( coords, coordIdxs, cv::Size( texWidth, texHeight ), idxMapMat );

    py::array_t<float> idxMap = py::array_t<float>( texHeight * texWidth * 3 );
    auto idxMapBuf = idxMap.request();
    memcpy( (float*)idxMapBuf.ptr, idxMapMat.ptr(), texHeight * texWidth * 3 * sizeof(float) );
    idxMap.resize( { texHeight, texWidth, 3 } );

    return idxMap;
}

void unwarpTex( const cv::Mat& img, const Eigen::Array3Xf& ptUVWs, const Eigen::Array3Xi& posIdxs, const cv::Mat& indexMap, cv::Mat& tex )
{
    cv::Size imgSize = img.size();
    cv::Size texSize = indexMap.size();

    tex.create( texSize, CV_8UC4 );
    tex.setTo( cv::Vec4b( 0, 0, 0, 0 ) );

    for( int v=0; v<texSize.height; v++ ){
        for( int u=0; u<texSize.width; u++ ){
            const cv::Vec3f& idx = indexMap.at<cv::Vec3f>( v, u );
            float lambda_1 = idx[ 1 ], lambda_2 = idx[ 2 ];
            float lambda_3 = 1.f - lambda_1 - lambda_2;

            int triId = int( idx[ 0 ] );
            const Eigen::Vector3i& posIdx = posIdxs.col( triId );

            Eigen::Vector3f triUVWs[ 3 ];
            for( int i=0; i<3; i++ ) triUVWs[ i ] = ptUVWs.col( posIdx[ i ] );

            Eigen::Vector3f pixUVW = triUVWs[ 0 ] * lambda_1 + triUVWs[ 1 ] * lambda_2 + triUVWs[ 2 ] * lambda_3;

            float x = pixUVW[ 0 ], y = pixUVW[ 1 ];
            int x0 = int( x ), y0 = int( y );
            int x1 = x0 + 1, y1 = y0 + 1;
            float xWeight = x - float( x0 ), yWeight = y - float( y0 );

            if( x0 < 0 || x1 >= imgSize.width || y0 < 0 || y1 >= imgSize.height ) continue;

            cv::Vec3f pixes[ 4 ];
            pixes[ 0 ] = (cv::Vec3f)img.at<cv::Vec3b>( y0, x0 );
            pixes[ 1 ] = (cv::Vec3f)img.at<cv::Vec3b>( y0, x1 );
            pixes[ 2 ] = (cv::Vec3f)img.at<cv::Vec3b>( y1, x0 );
            pixes[ 3 ] = (cv::Vec3f)img.at<cv::Vec3b>( y1, x1 );

            cv::Vec3f pixVal =
                    ( pixes[ 0 ] * ( 1.f - xWeight ) + pixes[ 1 ] * xWeight ) * ( 1.f - yWeight )
                    + ( pixes[ 2 ] * ( 1.f - xWeight ) + pixes[ 3 ] * xWeight ) * yWeight;

            for( int i=0; i<3; i++ )
                pixVal[ i ] = std::min( 255.f, std::max( 0.f, pixVal[ i ] ) );

            tex.at<cv::Vec4b>( v, u ) = cv::Vec4b( (uchar)pixVal[0], (uchar)pixVal[1], (uchar)pixVal[2], 255 );
        }
    }
}

std::vector<uchar> unwarpTex_py( const std::vector<uchar>& img_vec, int imgWidth, int imgHeight,
    const Eigen::Array3Xf& ptUVWs, const Eigen::Array3Xi& posIdxs,
    const std::vector<float>& idxMap_vec, int texWidth, int texHeight )
{
    cv::Mat img;
    img.create( imgHeight, imgWidth, CV_8UC3 );
    memcpy( img.ptr(), &img_vec[ 0 ], imgHeight * imgWidth * 3 );

    cv::Mat idxMap;
    idxMap.create( texHeight, texWidth, CV_32FC3 );
    memcpy( idxMap.ptr(), &idxMap_vec[ 0 ], texWidth * texHeight * 3 * sizeof(float) );

    cv::Mat tex;
    unwarpTex( img, ptUVWs, posIdxs, idxMap, tex );

    std::vector<uchar> tex_vec;
    tex_vec.resize( texWidth * texHeight * 4 );
    memcpy( &tex_vec[ 0 ], tex.ptr(), texWidth * texHeight * 4 );

    return tex_vec;
}

void unwarpTexNor( const cv::Mat& img, const Eigen::Array3Xf& ptUVWs, const Eigen::Array3Xi& posIdxs,
                   const cv::Mat& indexMap, const cv::Mat& texMask, cv::Mat& tex )
{
    // Calculate the ptUVWNors
    int ptNum = int( ptUVWs.cols() ), triNum = int( posIdxs.cols() );
    Eigen::Array3Xf ptUVWNors( 3, ptNum );
    ptUVWNors.setZero();
    for( int triId=0; triId<triNum; triId++ ){
        const Eigen::Vector3i& posIdx = posIdxs.col( triId );
        Eigen::Vector3f e0 = ptUVWs.col( posIdx[ 1 ] ) - ptUVWs.col( posIdx[ 0 ] );
        Eigen::Vector3f e1 = ptUVWs.col( posIdx[ 2 ] ) - ptUVWs.col( posIdx[ 0 ] );
        Eigen::Vector3f nor = e0.cross( e1 );

        for( int i=0; i<3; i++ ){
            Eigen::Vector3f ptNor = ptUVWNors.col( posIdx[ i ] );
            ptUVWNors.col( posIdx[ i ] ) = ptNor + nor;
        }
    }
    for( int ptId=0; ptId<ptNum; ptId++ ){
        Eigen::Vector3f ptNor = ptUVWNors.col( ptId );
        ptUVWNors.col( ptId ) = ptNor.normalized();
    }

    // Unwarp texture
    cv::Size imgSize = img.size();
    cv::Size texSize = indexMap.size();

    tex.create( texSize, CV_8UC4 );
    tex.setTo( cv::Vec4b( 0, 0, 0, 0 ) );

    const float cPi = 3.1415926535897932384626433f;

    #pragma omp parallel for
    for( int v=0; v<texSize.height; v++ ){
        for( int u=0; u<texSize.width; u++ ){
            float pixWeight = texMask.at<float>( v, u );
            if( pixWeight == 0.f ) continue;

            const cv::Vec3f& idx = indexMap.at<cv::Vec3f>( v, u );
            float lambda_1 = idx[ 1 ], lambda_2 = idx[ 2 ];
            float lambda_3 = 1.f - lambda_1 - lambda_2;

            int triId = int( idx[ 0 ] );
            const Eigen::Vector3i& posIdx = posIdxs.col( triId );

            Eigen::Vector3f triUVWs[ 3 ];
            for( int i=0; i<3; i++ ) triUVWs[ i ] = ptUVWs.col( posIdx[ i ] );

            Eigen::Vector3f pixUVW = triUVWs[ 0 ] * lambda_1 + triUVWs[ 1 ] * lambda_2 + triUVWs[ 2 ] * lambda_3;

            float x = pixUVW[ 0 ], y = pixUVW[ 1 ];
            int x0 = int( x ), y0 = int( y );
            int x1 = x0 + 1, y1 = y0 + 1;
            float xWeight = x - float( x0 ), yWeight = y - float( y0 );

            if( x0 < 0 || x1 >= imgSize.width || y0 < 0 || y1 >= imgSize.height ) continue;

            cv::Vec3f pixes[ 4 ];
            pixes[ 0 ] = (cv::Vec3f)img.at<cv::Vec3b>( y0, x0 );
            pixes[ 1 ] = (cv::Vec3f)img.at<cv::Vec3b>( y0, x1 );
            pixes[ 2 ] = (cv::Vec3f)img.at<cv::Vec3b>( y1, x0 );
            pixes[ 3 ] = (cv::Vec3f)img.at<cv::Vec3b>( y1, x1 );

            cv::Vec3f pixVal =
                    ( pixes[ 0 ] * ( 1.f - xWeight ) + pixes[ 1 ] * xWeight ) * ( 1.f - yWeight )
                    + ( pixes[ 2 ] * ( 1.f - xWeight ) + pixes[ 3 ] * xWeight ) * yWeight;

            // Get the pixel's normal
            Eigen::Vector3f triNors[ 3 ];
            for( int i=0; i<3; i++ ) triNors[ i ] = ptUVWNors.col( posIdx[ i ] );
            Eigen::Vector3f pixNor = triNors[ 0 ] * lambda_1 + triNors[ 1 ] * lambda_2 + triNors[ 2 ] * lambda_3;

            if( pixNor[ 2 ] < 0.f ) pixWeight *= tanh( pixNor[ 2 ] * 2.f * cPi + cPi );
            pixVal *= pixWeight;

            // Set pixel
            for( int i=0; i<3; i++ )
                pixVal[ i ] = std::min( 255.f, std::max( 0.f, pixVal[ i ] ) );
            float alpha = ( pixNor[ 2 ] + 1.f ) * 0.5f;
            alpha = std::min( 255.f, std::max( 0.f, alpha * 255.f ) );
            tex.at<cv::Vec4b>( v, u ) = cv::Vec4b( (uchar)pixVal[0], (uchar)pixVal[1], (uchar)pixVal[2], (uchar)alpha );
        }
    }
}

py::array_t<uchar> unwarpTexNor_py( py::array_t<uchar>& py_imgs, py::array_t<float>& py_ptUVWs,
                                    py::array_t<int>& py_posIdxs, py::array_t<float>& py_idxMap,
                                    py::array_t<int>& py_viewIds, py::array_t<float>& py_texMasks,
                                    py::array_t<int>& py_texRect, py::array_t<int>& py_texSize )
{
    // Get the buffer, and get the parameters
    auto py_imgsBuf = py_imgs.request();
    auto py_ptUVWsBuf = py_ptUVWs.request();
    auto py_posIdxsBuf = py_posIdxs.request();
    auto py_idxMapBuf = py_idxMap.request();
    auto py_viewIdsBuf = py_viewIds.request();
    auto py_texMasksBuf = py_texMasks.request();
    auto py_texRectBuf = py_texRect.request();
    auto py_texSizeBuf = py_texSize.request();

    int imgNum = int( py_imgsBuf.shape[ 0 ] );
    int imgHeight = int( py_imgsBuf.shape[ 1 ] ), imgWidth = int( py_imgsBuf.shape[ 2 ] );
    int texHeight = int( py_idxMapBuf.shape[ 0 ] ), texWidth = int( py_idxMapBuf.shape[ 1 ] );
    int ptNum = int( py_ptUVWsBuf.shape[ 1 ] );
    int triNum = int( py_posIdxsBuf.shape[ 0 ] );
    int viewNum = int( py_texMasksBuf.shape[ 0 ] );

    // Build the data
    cv::Mat idxMap;
    idxMap.create( texHeight, texWidth, CV_32FC3 );
    memcpy( idxMap.ptr(), (float*)py_idxMapBuf.ptr, texWidth * texHeight * 3 * sizeof(float) );

    Eigen::Array3Xi posIdxs( 3, triNum );
    memcpy( posIdxs.data(), (int*)py_posIdxsBuf.ptr, triNum * 3 * sizeof(int) );

    float* py_texMaksPtr = (float*)py_texMasksBuf.ptr;
    std::vector<cv::Mat> texMasks( viewNum );
    for( int viewId=0; viewId<viewNum; viewId++ ){
        cv::Mat& texMask = texMasks[ viewId ];
        texMask.create( texHeight, texWidth, CV_32F );
        memcpy( texMask.ptr(), &py_texMaksPtr[ texHeight * texWidth * viewId ],
                texHeight * texWidth * sizeof( float ) );
        cv::flip( texMask, texMask, 0 );
    }

    // Unwarp
    cv::Rect cropRect( 0, 0, texHeight, texWidth );
    if( py_texRectBuf.shape[ 0 ] == 4 ){
        int* ptr = (int*)py_texRectBuf.ptr;
        cropRect = cv::Rect( ptr[0], ptr[1], ptr[2], ptr[3] );
    }

    cv::Size cropSize( texHeight, texWidth );
    if( py_texSizeBuf.shape[ 0 ] == 2 ){
        int* ptr = (int*)py_texSizeBuf.ptr;
        cropSize = cv::Size( ptr[ 0 ], ptr[ 1 ] );
    }

    py::array_t<uchar> py_texes = py::array_t<uchar>( imgNum * cropSize.width * cropSize.height * 4 );
    auto py_texesBuf = py_texes.request();
    uchar* py_texesPtr = (uchar*)py_texesBuf.ptr;

    uchar* py_imgsPtr = (uchar*)py_imgsBuf.ptr;
    float* py_ptUVWsPtr = (float*)py_ptUVWsBuf.ptr;
    int* py_viewIdsPtr = (int*)py_viewIdsBuf.ptr;
    for( int imgId=0; imgId<imgNum; imgId++ ){
        cv::Mat img;
        img.create( imgHeight, imgWidth, CV_8UC3 );
        memcpy( img.ptr(), &py_imgsPtr[ imgHeight * imgWidth * 3 * imgId ], imgHeight * imgWidth * 3 );

        Eigen::Array3Xf ptUVWs( 3, ptNum );
        memcpy( ptUVWs.data(), &py_ptUVWsPtr[ ptNum * 3 * imgId ], ptNum * 3 * sizeof( float ) );

        cv::Mat tex;
        const cv::Mat& texMask = texMasks[ py_viewIdsPtr[ imgId ] ];
        unwarpTexNor( img, ptUVWs, posIdxs, idxMap, texMask, tex );
        cv::flip( tex, tex, 0 );
        cv::resize( tex( cropRect ), tex, cropSize );

        memcpy( &py_texesPtr[ cropSize.width * cropSize.height * 4 * imgId ], tex.ptr(),
                cropSize.width * cropSize.height * 4 );

        printf( " Unwarp texture for image %d/%d\r", imgId + 1, imgNum );
    }
    printf( "\n" );

    py_texes.resize( { imgNum, cropSize.height, cropSize.width, 4 } );
    return py_texes;
}

void unwarpPosMap( const Eigen::Array3Xf& ptXYZs, const Eigen::Array3Xi& posIdxs,
                const cv::Mat& indexMap, cv::Mat& posMap  )
{
    cv::Size texSize = indexMap.size();
    posMap.create( texSize, CV_32FC3 );
    posMap.setTo( cv::Vec3f( 0.0, 0.0, 0.0 ) );

    for( int v=0; v<texSize.height; v++ ){
        for( int u=0; u<texSize.width; u++ ){
            const cv::Vec3f& idx = indexMap.at<cv::Vec3f>( v, u );
            int triId = int( idx[ 0 ] );
            float lambda_1 = idx[ 1 ], lambda_2 = idx[ 2 ];
            float lambda_3 = 1.f - lambda_1 - lambda_2;

            const Eigen::Vector3i& posIdx = posIdxs.col( triId );
            Eigen::Vector3f triXYZs[ 3 ];
            for( int i=0; i<3; i++ ) triXYZs[ i ] = ptXYZs.col( posIdx[ i ] );

            Eigen::Vector3f pixXYZ = triXYZs[ 0 ] * lambda_1
                                    + triXYZs[ 1 ] * lambda_2
                                    + triXYZs[ 2 ] * lambda_3;

            posMap.at<cv::Vec3f>( v, u ) = cv::Vec3f( pixXYZ[ 0 ], pixXYZ[ 1 ], pixXYZ[ 2 ] );
        }
    }
}

std::vector<float> unwarpPosMap_py( const Eigen::Array3Xf& ptXYZs, const Eigen::Array3Xi& posIdxs,
                const std::vector<float>& idxMap_vec, int texWidth, int texHeight )
{
    cv::Mat idxMap;
    idxMap.create( texHeight, texWidth, CV_32FC3 );
    memcpy( idxMap.ptr(), &idxMap_vec[ 0 ], texWidth * texHeight * 3 * sizeof(float) );

    cv::Mat posMap;
    unwarpPosMap( ptXYZs, posIdxs, idxMap, posMap );

    std::vector<float> posMap_vec;
    posMap_vec.resize( texWidth * texHeight * 3 );
    memcpy( &posMap_vec[ 0 ], posMap.ptr(), texWidth * texHeight * 3 * sizeof( float ) );

    return posMap_vec;
}


// --------------------------------------------------
// *Pybind
PYBIND11_MODULE( texUnwarp, m ){
    m.doc() = "Unwarp texture from image";

    m.def( "calcIndexMap", &calcIndexMap_py, "Calculate texture index map" );
    m.def( "unwarpTex", &unwarpTex_py, "Unwarp the texture" );
    m.def( "unwarpTexNor", &unwarpTexNor_py, "Unwarp the texture" );
    m.def( "unwarpPos", &unwarpPosMap_py, "Unwarp the position map" );
}
