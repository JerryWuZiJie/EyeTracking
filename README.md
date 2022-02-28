# EyeTracking

* This project is for my DP2 courses working with Professor Ivan Selesnick.

* The goal of the project is to modify an algorithm proposed by [this](https://jov.arvojournals.org/article.aspx?articleid=2772700) paper.
  * The algorithm is used to denoise the time series data of eye movement. The algorithm, in combine with simple VT (velocity thresholding), is more accurate in detecting both normal and slow saccades than other algorithms.
  * The algorithm is currently not suitable for subjects with nystagmus or for eye-tracking data with smooth pursuit eye movements, which this project is trying to improve on.
  * In the future, we may also try to add the functionality to identify anti-saccade.


* The original code can be found [here](https://eeweb.engineering.nyu.edu/iselesni/eye-movement/).