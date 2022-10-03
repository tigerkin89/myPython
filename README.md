
# 파이썬
* [쥬피터 랩 CentOS 데몬 실행하기] (https://nomad-programmer.tistory.com/26)
* [Python Regular Expression Quick Guide] (https://www.py4e.com/lectures3/Pythonlearn-11-Regex-Handout.txt)
* [ Google PageRank Algorithm] (http://infolab.stanford.edu/~backrub/google.html)
# 데이터 과학
* [50 years of Data Science, David Donoho] (http://courses.csail.mit.edu/18.337/2015/docs/50YearsDataScience.pdf)
* [Science Isn't Broken: p-hacking] (http://fivethirtyeight.com/features/science-isnt-broken/)
* [Goodhart's Law] (https://dataskeptic.com/blog/episodes/2016/goodharts-law)
* [Data Scientists, The 5 Graph Algorithms that you should know] (https://towardsdatascience.com/data-scientists-the-five-graph-algorithms-that-you-should-know-30f454fa5513)
* 윤리에 대하여
  * http://approximatelycorrect.com/2016/11/07/the-foundations-of-algorithmic-bias/

# 머신러닝
* [scikit-learn Homepage] (http://scikit-learn.org/)
* [scikit-get start] (https://scikit-learn.org/stable/getting_started.html#)
* [scikit-learn User Guide]  (http://scikit-learn.org/stable/user_guide.html)
* [scikit-learn API reference]  (http://scikit-learn.org/stable/modules/classes.html)
* [SciPy] (http://www.scipy.org)

# 파이썬 플로팅
* Visual Studio Code 환경설정 (graph 출력하기)
  * Jupyter Notebook Renderes 설치
  * 셀에서 %matplotlib widget 실행
* [Dark Horse Analytics] (http://www.darkhorseanalytics.com/)
* [Useful Junk?: The Effects of Visual Embellishment on Comprehension and Memorability of Charts] (http://www.stat.columbia.edu/~gelman/communication/Bateman2010.pdf)
* [Graphics Lies, Misleading Visuals] (https://faculty.ucmerced.edu/jvevea/classes/Spark/readings/Cairo2015_Chapter_GraphicsLiesMisleadingVisuals.pdf)
* [matplotlib] (http://www.aosabook.org/en/matplotlib.html)
* [The Architecture of Open Source Applications, Volume II: Structure, Scale, and a Few More Fearless Hacks (Vol. 2)] (https://archive.org/download/aosa_v2/aosa_v2.pdf)
* (Ten Simple Rules for Better Figures) [https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833]
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
