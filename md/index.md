<center>
<span style="color: #ff0000"><b>乗るしかないこのビッグウェーブに</b></span>
</center>

Deep Learning（深層学習）に関連するまとめページとして使用する予定です。Deep Learningに関する記事・スライド・論文・動画・書籍へのリンクをまとめています。最新の研究動向は全然把握できていないので今後研究を進めるなかで記録していきたいと思います。読んだ論文の概要も簡単にまとめていく予定です。

本ブログでは、当面の間、Theanoを使ったDeep Learningアルゴリズムの実装に関してまとめていきたいと思います！

# 関連記事

- [パターン認識と機械学習（PRML）まとめ](http://aidiary.hatenablog.com/entry/20100829/1283068351)（2010/8/29） - PRMLの機械学習アルゴリズムをPythonで実装
- [Pythonによるモンテカルロ法入門](http://aidiary.hatenablog.com/entry/20140620/1403272044)（2014/6/20） - MCMCまでたどり着いてないので途半端だけど
- [人工知能を実現する学習アルゴリズムに必要な能力](http://aidiary.hatenablog.com/entry/20140910/1410352095)（2014/9/10） - Bengioさんの論文から
- [TheanoをWindowsにインストール](http://aidiary.hatenablog.com/entry/20150127/1422364425)（2015/1/27） - Windows環境を整備、GPUで高速化まで

以下は執筆予定...

- TheanoによるLogistic Regressionの実装
- TheanoによるMultilayer Perceptronの実装
- TheanoによるConvolutional Neural Networkの実装
- TheanoによるAutoencoderの実装
- TheanoによるRestricted Boltzmann Machineの実装
- TheanoによるDeep Belief Networkの実装

# ライブラリ

今はTheanoを使った実装を勉強中。そのうちPylearn2に移行するかも。

- [DeepLearning.net](http://deeplearning.net/) - Deep Learning情報の総本山、チュートリアル、論文リストなど
- [Theano](http://www.deeplearning.net/software/theano/)
- [Theano Tutorial](http://deeplearning.net/software/theano/tutorial/) - Theanoのチュートリアル
- [Deep Learning Tutorial](http://deeplearning.net/tutorial/contents.html)  - Theanoを用いたDeep Learningアルゴリズム実装の解説
- [Pylearn2](http://deeplearning.net/software/pylearn2/) - Theanoを使ったDeep Learningライブラリ
- [Caffe](http://caffe.berkeleyvision.org/)
- H2O - Rパッケージ？
- Torch7 - Facebook

# 論文

## 人工知能学会の特集

サーベイのとっかかりとして最適だけれど、残念ながら無料ダウンロードはできない。1年経ったら全体公開してほしいな。私のブックマークは参考にさせていただきました。

- [連載解説「Deep Learning（深層学習）」にあたって](http://ci.nii.ac.jp/naid/110009604084)
- [第1回 ディープボルツマンマシン入門 : ボルツマンマシン学習の基礎](http://ci.nii.ac.jp/naid/110009604085)
- [第2回 多層ニューラルネットワークによる深層表現の学習](http://ci.nii.ac.jp/naid/110009615748)
- [第3回 大規模Deep Learning(深層学習)の実現技術](http://ci.nii.ac.jp/naid/110009636279)
- [第4回 画像認識のための深層学習](http://ci.nii.ac.jp/naid/110009675090)
- [第5回 音声認識のための深層学習](http://ci.nii.ac.jp/naid/110009810077)
- [第6回 自然言語処理のための深層学習](http://ci.nii.ac.jp/naid/110009816909)
- [第7回 コントラスティブダイバージェンス法とその周辺](http://ci.nii.ac.jp/naid/40020149308)
- [私のブックマーク Deep Learning](http://ci.nii.ac.jp/naid/110009832022)

## 総説

- [Deep Learning Reading List](http://deeplearning.net/reading-list/) - 読むべき論文のリスト
- [DEEP LEARNING](http://www.iro.umontreal.ca/~bengioy/dlbook/) - Bengioさんの本（PDFで無料公開）
- [Learning Deep Architectures for AI](http://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)
- [An Introduction to Restricted Boltzmann Machines ](http://image.diku.dk/igel/paper/AItRBM-proof.pdf)
- [Ng's Lecture Note: Sparse Autoencoder](http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
- [Unsupervised Feature Learning and Deep Learning Tutorial](http://deeplearning.stanford.edu/tutorial/)
- [A Practical Guide to Training Restricted Boltzmann Machine](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
- [Deep Learning: Methods and Applications](http://research.microsoft.com/pubs/209355/DeepLearning-NowPublishing-Vol7-SIG-039.pdf)

## 国際会議のチュートリアル

- [Representation learning tutorial](http://www.iro.umontreal.ca/~bengioy/talks/deep-learning-gss2012.html) - ICML2012のチュートリアル
- [Deep learning methods for vision](http://cs.nyu.edu/~fergus/tutorials/deep_learning_cvpr12/) - CVPR2012のチュートリアル
- [Deep learning for NLP (without Magic)](http://www.socher.org/index.php/DeepLearningTutorial/DeepLearningTutorial) - ACL2012のチュートリアル
- [Deep learning and its applications in signal processing](http://www.icassp2012.com/Tutorial_09.asp) - ICASSP2012のチュートリアル
- [Graduate summer school: Deep learning, feature learning](http://www.ipam.ucla.edu/programs/summer-schools/graduate-summer-school-deep-learning-feature-learning/) - Hintonさんのサマースクール

## コンピュータビジョン関連

- [Building High-level Features Using Large Scale Unsupervised Learning](http://static.googleusercontent.com/media/research.google.com/ja//archive/unsupervised_icml2012.pdf) - Googleの猫論文
- [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
- [The Shape Boltzmann Machine: A strong model of object shape](http://arkitus.com/files/cvpr-12-eslami-sbm.pdf)

## 音声関連

- [Deep Neural Networks for Acoustic Modeling in Speech Recognition](http://research.microsoft.com/pubs/171498/HintonDengYuEtAl-SPM2012.pdf) - 音声認識サーベイ
- Deep Learning for Acoustic Modeling in Parametric Speech Generation - 音声合成サーベイ

## 自然言語処理関連

- [Linguistic Regularities in Continuous Space Word Representations](http://aclweb.org/anthology/N/N13/N13-1090.pdf) - word2vec, King-Man+Woman=Queen

## 強化学習関連論文

個人的に一番興味がある分野。

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) - DQN
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) - DQNのNature論文、コードは[ここ](https://sites.google.com/a/deepmind.com/dqn/)で公開

# ブログ記事など

はてブで記録していたものをまとめてみました。これ以外に面白いと思った記事がたくさんあったと思うのであとで追加していきたいと思います。もしおすすめの面白い記事がありましたらコメント欄で教えてください。個人的に「実装しました」系がすごく好きです。

## 解説

- [ディープラーニングチュートリアル（もしくは研究動向報告）](http://www.vision.is.tohoku.ac.jp/files/9313/6601/7876/CVIM_tutorial_deep_learning.pdf)
- [ニューラルネットの逆襲](http://research.preferred.jp/2012/11/deep-learning/)
- [一般向けのDeep Learning](http://www.slideshare.net/pfi/deep-learning-22350063)
- [Deep Learning技術の今](http://www.slideshare.net/beam2d/deep-learning20140130)
- [Deep Learning : Bengio先生のおすすめレシピ](http://conditional.github.io/blog/2013/09/22/practical-recommendations-for-gradient-based-training-of-deep-architectures/)
- [はじめるDeep learning](http://qiita.com/icoxfog417/items/65e800c3a2094457c3a0)

## Theano

- [Theano入門](http://www.chino-js.com/ja/tech/theano-rbm/)
- [Theano解説](http://d.hatena.ne.jp/saket/20121207/1354867911)
- [TheanoでDeep Learning](http://sinhrks.hatenablog.com/entry/2014/11/26/002818)
- [Deep Learningを実装する](http://www.slideshare.net/tushuhei/121227deep-learning-iitsuka)

## Caffe

- [ご注文はDeep Learningですか？](http://kivantium.hateblo.jp/entry/2015/02/20/214909)
- [Caffeで手軽に画像分類](http://techblog.yahoo.co.jp/programming/caffe-intro/)
- [CaffeでDeep Q-Networkを実装して深層強化学習してみた](http://d.hatena.ne.jp/muupan/20141021/1413850461)

# 動画

NgさんのCourseraの機械学習コースはDeep Learningこそ扱っていませんが、ロジスティック回帰やニューラルネットの技術を実践的に学べるためおすすめ。

- [Machine Learning](https://www.coursera.org/learn/machine-learning) - Ngさんによる機械学習講義
- [Neural Networks for Machine Learning](https://www.coursera.org/course/neuralnets) - Hintonさんによるニューラルネット講義
- [The Next Generation of Neural Networks](https://www.youtube.com/watch?v=AyzOUbkUf3M) - Google Talksでの講演
- [Recent Developments in Deep Learning](https://www.youtube.com/watch?v=VdIURAu1-aU) - Google Talksでの講演

# 書籍

松尾さんの本から入るのがおすすめ。深層学習の教科書はよくまとまっていてよかった。

- [asin:B00UAAK07S:detail]
- [asin:B00OH3ZU7Y:detail]
- [asin:B00UT1RJ7M:detail]
- [asin:B00SI0GIJ6:detail]
- [asin:4061529021:detail]
- [asin:4627852118:detail]
- [asin:4788514222:detail]
- [asin:B00AF1AYTQ:detail]
