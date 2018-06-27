#pragma once
class FaceRecognition
{
public:
    FaceRecognition();
    ~FaceRecognition();

    void SimpleApproach();
	void Explore();
	void EigenFaces_explore();
	void PCAImage(cv::Mat img, cv::Mat face_C, cv::Mat face_U, cv::Mat eigen_vectors_L, cv::Mat col_picked_estimate, int idx);


	void EigenFaces_training();
	void EigenFaces_test();
	std::vector<std::string> TraverseFiles();
	void find_file(char* lpPath, std::vector< std::string> &fileList);
	std::vector<double> PCAImageCoeffs(int face_C_W, cv::Mat face_U, cv::Mat eigen_vectors_L,  int idx);
	double MahalanobisDistance(std::vector<double> coeff, std::vector<double> coeff1, std::vector<double> lambdas);
	int m_principleNum = 20;

	void SaveImageMatrixToFile();
	int ReshapeImageAndGetName( std::vector<std::string>& people_names, std::vector<int>& people_names_numeral, cv::Mat& face_U);
};

