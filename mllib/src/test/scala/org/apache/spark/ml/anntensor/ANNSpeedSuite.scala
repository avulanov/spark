package org.apache.spark.ml.anntensor

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classificationtensor.{MultilayerPerceptronClassifier => TMLP}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.{MLUtils, MLlibTestSparkContext}

class ANNSpeedSuite extends SparkFunSuite with MLlibTestSparkContext {

//  test ("speed test") {
//    val mnistPath = System.getenv("MNIST_HOME")
//    println(mnistPath + "/mnist.scale")
//    val dataFrame = sqlContext.
//      createDataFrame(MLUtils.loadLibSVMFile(sc, mnistPath + "/mnist.scale", 784)).persist()
//    dataFrame.count()
//    val mlp = new MultilayerPerceptronClassifier().setLayers(Array(784, 32, 10))
//      .setTol(10e-9)
//      .setMaxIter(20)
//      .setSeed(1234L)
//    val t = System.nanoTime()
//    val model = mlp.fit(dataFrame)
//    val total = System.nanoTime() - t
//    println("Total time: " + total / 1e9 + " s. (should be ~42s. without native BLAS")
//    val test = sqlContext.
//      createDataFrame(MLUtils.loadLibSVMFile(sc, mnistPath + "/mnist.scale.t", 784)).persist()
//    test.count()
//    val result = model.transform(test)
//    val pl = result.select("prediction", "label")
//    val ev = new MulticlassClassificationEvaluator().setMetricName("precision")
//    println("Accuracy: " + ev.evaluate(pl))
//  }

  test ("speed test with tensor") {
    val mnistPath = System.getenv("MNIST_HOME")
    println(mnistPath + "/mnist.scale")
    val dataFrame = sqlContext.
      createDataFrame(MLUtils.loadLibSVMFile(sc, mnistPath + "/mnist.scale", 784)).persist()
    dataFrame.count()
    val warmUp = new MultilayerPerceptronClassifier().setLayers(Array(784, 32, 10))
      .setTol(10e-9)
      .setMaxIter(2)
      .setSeed(1234L)
      .fit(dataFrame)

    val mlp = new MultilayerPerceptronClassifier().setLayers(Array(784, 32, 10))
      .setTol(10e-9)
      .setMaxIter(20)
      .setSeed(1234L)
    val t = System.nanoTime()
    val model = mlp.fit(dataFrame)
    val total = System.nanoTime() - t
    println("ANN total time: " + total / 1e9 +
      " s. (should be ~37. without native BLAS with warm-up)")
    val tensorMLP = new TMLP().setLayers(Array(784, 32, 10))
      .setTol(10e-9)
      .setMaxIter(20)
      .setSeed(1234L)
    val tTensor = System.nanoTime()
    val tModel = tensorMLP.fit(dataFrame)
    val totalTensor = System.nanoTime() - tTensor
    println("Tensor total time: " + totalTensor / 1e9 +
      " s. (should be ~37s. without native BLAS with warm-up)")

    val test = sqlContext.
      createDataFrame(MLUtils.loadLibSVMFile(sc, mnistPath + "/mnist.scale.t", 784)).persist()
    test.count()
    val result = model.transform(test)
    val pl = result.select("prediction", "label")
    val ev = new MulticlassClassificationEvaluator().setMetricName("precision")
    println("ANN Accuracy: " + ev.evaluate(pl))
    val tResult = tModel.transform(test)
    val tpl = tResult.select("prediction", "label")
    val tev = new MulticlassClassificationEvaluator().setMetricName("precision")
    println("Tensor Accuracy: " + tev.evaluate(tpl))

  }

}
