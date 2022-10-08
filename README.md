
# 파이썬
* [쥬피터 랩 CentOS 데몬 실행하기](https://nomad-programmer.tistory.com/26)
* [Python Regular Expression Quick Guide](https://www.py4e.com/lectures3/Pythonlearn-11-Regex-Handout.txt)
* [ Google PageRank Algorithm](http://infolab.stanford.edu/~backrub/google.html)
# 데이터 과학
* [50 years of Data Science, David Donoho](http://courses.csail.mit.edu/18.337/2015/docs/50YearsDataScience.pdf)
* [Science Isn't Broken: p-hacking](http://fivethirtyeight.com/features/science-isnt-broken/)
* [Goodhart's Law](https://dataskeptic.com/blog/episodes/2016/goodharts-law)
* [Data Scientists, The 5 Graph Algorithms that you should know](https://towardsdatascience.com/data-scientists-the-five-graph-algorithms-that-you-should-know-30f454fa5513)
* 윤리에 대하여
  * http://approximatelycorrect.com/2016/11/07/the-foundations-of-algorithmic-bias/

# 머신러닝
* [a few useful things to Know about machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
* [scikit vs tensor flow](https://stackoverflow.com/questions/61233004/whats-the-difference-between-scikit-learn-and-tensorflow-is-it-possible-to-use)
* [scikit-learn Homepage](http://scikit-learn.org/)
* [scikit-get start](https://scikit-learn.org/stable/getting_started.html#)
* [scikit-learn User Guide](http://scikit-learn.org/stable/user_guide.html)
* [scikit-learn API reference](http://scikit-learn.org/stable/modules/classes.html)
* [scikit-learn Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
* [SciPy](http://www.scipy.org)
* Represent / Train / Evaluate / Refine Cycle
  * [모델링 과정](https://bigdaheta.tistory.com/54)
  * Representation
    * Extract and select object features
    * [scikit-learn 데이터셋(dataset) 다루기](https://teddylee777.github.io/scikit-learn/scikit-learn-dataset)
    * [scikit-learn 데이터 전처리](https://teddylee777.github.io/scikit-learn/scikit-learn-preprocessing)
    * [Pandas와 scikit-learn으로 정말 간단한 pre-processing 몇 가지 팁](https://teddylee777.github.io/scikit-learn/sklearn%EC%99%80-pandas%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B0%84%EB%8B%A8-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D)
    
  * Train Model 
    * Fit the estimator to the data
    * [train_test_split 모듈을 활용하여 학습과 테스트 세트 분리](https://teddylee777.github.io/scikit-learn/train-test-split)
    * [sklearn의traintest_split의 randomstate의 필요성](https://intrepidgeeks.com/tutorial/init-state-split)
    * [klearn의 K-Nearest Neighbors 분류기를 활용하여 Iris 꽃 종류 분류하기 (Classifier)](https://teddylee777.github.io/scikit-learn/KNeighborsClassifier%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%B6%84%EB%A5%98%EA%B8%B0-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0)
  * Evaluation
  * Feature and model refinement
  
# 시계열 분석
* ARIMA 모델
  * [ARIMA 모델](https://velog.io/@sjina0722/시계열분석-ARIMA-모델)
  * [ARIMA 모델 동영상) [https://www.youtube.com/watch?v=ma_L2YRWMHI]
* Prophet 모델
* LSTM
  * Long Short-Term Memory (LSTM) 이해하기 [https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr]
  * (Python : Keras : Lstm : 오존 예측 : 예제, 사용법,활용법) https://jjeongil.tistory.com/955

# 파이썬 플로팅
* Visual Studio Code 환경설정 (graph 출력하기)
  * Jupyter Notebook Renderes 설치
  * 셀에서 %matplotlib widget 실행
* [Dark Horse Analytics](http://www.darkhorseanalytics.com/)
* [Useful Junk?: The Effects of Visual Embellishment on Comprehension and Memorability of Charts](http://www.stat.columbia.edu/~gelman/communication/Bateman2010.pdf)
* [Graphics Lies, Misleading Visuals] (https://faculty.ucmerced.edu/jvevea/classes/Spark/readings/Cairo2015_Chapter_GraphicsLiesMisleadingVisuals.pdf)
* [matplotlib](http://www.aosabook.org/en/matplotlib.html)
* [The Architecture of Open Source Applications, Volume II: Structure, Scale, and a Few More Fearless Hacks (Vol. 2)] (https://archive.org/download/aosa_v2/aosa_v2.pdf)
* (Ten Simple Rules for Better Figures)[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833]
* 에러 대처
  * module 'pandas' has no attribute 'tools' 에러 대처
    * pd.tools.plotting.scatter_matrix(iris); 과 같이 pd.tools인 경우 발생
    * 다음과 같이 처리함
      * from pandas.plotting import parallel_coordinates
      * pd.plotting.scatter_matrix(iris);
  * 'Rectangle' object has no property 'normed'
    * plt.hist([v1, v2], histtype='barstacked', normed=True);
    * normed 대신 density를 사용하고, stacked=True 추가
      * plt.hist([v1, v2], histtype='barstacked', density=True, stacked=True);
# 파이썬 유용한팁
* [진행시간, 진행율 표시](https://jimmy-ai.tistory.com/13)
```python
from tqdm import tqdm

result = df.apply(function, axis = 1) # tqdm 미적용
result = df.progress_apply(function, axis = 1) # tqdm 적용
 
j = 0
for i in tqdm(range(10000000)):
    j += 1
```
* 판다스 출력 옵션
```python
warnings.filterwarnings(action='ignore') # warning ignore 

pd.options.display.max_rows = 80
pd.options.display.max_columns = 80
```
