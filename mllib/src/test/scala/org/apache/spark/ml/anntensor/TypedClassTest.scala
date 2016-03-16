package org.apache.spark.ml.anntensor

object Math {
  trait NumberLike[@specialized (Double, Int) T] {
    def plus(x: T, y: T): T
  }
  object NumberLike {
    implicit object NumberLikeDouble extends NumberLike[Double] {
      def plus(x: Double, y: Double): Double = x + y
    }
    implicit object NumberLikeInt extends NumberLike[Int] {
      def plus(x: Int, y: Int): Int = x + y
    }
  }
}
object Statistics {
  import Math.NumberLike
  def plus[@specialized (Double, Int) T](x: T, y: T)(implicit ev: NumberLike[T]): T =
    ev.plus(x, y)
  def plusDouble(x: Double, y: Double): Double = x + y
}
import Math.NumberLike
class My[@specialized (Double, Int) T](implicit ev: NumberLike[T]) {
  def plus(x: T, y: T): T = ev.plus(x, y)
}

object TypedClassTest {
  def main(args: Array[String]): Unit = {
//    Statistics.plus(2.0, 2.0)
//    Statistics.plusDouble(2.0, 2.0)
    val m = new My[Double]()
    m.plus(2.0, 2.0)
  }
}
