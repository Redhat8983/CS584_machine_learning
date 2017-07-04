Data use in each question:
1.1D 2-class Gaussian discriminant analysis
  1_mm.data
  1_iris.data

2.nD 2-class Gaussian discriminant analysis
  2_nD2C.data
  1_mm.data

3.nD m-class Gaussian discriminant analysis
  3class_iris.dat( 3class)
  pima-indians-diabetes.dat( 2class)

4.Naïve Bayes with Bernoulli features
  4_spe_heart.data

5.Naïve Bayes Binominal features
  5_monk.data **
  ** class put at first element for each line.

If need to use other data. replace in code

   def InitReadData(self):
     f = open('PUT DATA NAME HERE', 'r')
and use same stracture as original data set

** Put data and code in same directory
