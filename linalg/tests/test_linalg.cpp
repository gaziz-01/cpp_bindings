#include "LinearAlgebra.h"
#include <gtest/gtest.h>

TEST(LinearAlgebraTests, DotProductBlas) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  EXPECT_DOUBLE_EQ(32.0, LinearAlgebra::dotBlas(a, b));
}

TEST(LinearAlgebraTests, MatrixMultiplicationBlas) {
  std::vector<std::vector<double>> matrix_a = {{1.0, 2.0}, {3.0, 4.0}};
  std::vector<std::vector<double>> matrix_b = {{2.0, 0.0}, {1.0, 3.0}};
  auto result = LinearAlgebra::matmulBlas(matrix_a, matrix_b);
  EXPECT_DOUBLE_EQ(result[0][0], 4.0);
  EXPECT_DOUBLE_EQ(result[0][1], 6.0);
  EXPECT_DOUBLE_EQ(result[1][0], 10.0);
  EXPECT_DOUBLE_EQ(result[1][1], 12.0);
}

TEST(LinearAlgebraTests, CosineSimilarityBlas) {
  std::vector<std::vector<double>> tensor_a = {{1,2,3},{4,5,6}};
  std::vector<std::vector<double>> tensor_b = {{7,8,9},{10,11,12}};
  auto result = LinearAlgebra::cosine_similarity(tensor_a,tensor_b);

  // got these values as a result of torch.nn.functional.cosine_similarity
  EXPECT_DOUBLE_EQ(result[0], 0.9594119455666703); 
  EXPECT_DOUBLE_EQ(result[1], 0.9961498555841326);

}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
