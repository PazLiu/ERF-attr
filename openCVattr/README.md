1、本代码将用极端随机森林（Extremely Random Forests， ERF）来训练图像分类器。一个目标识
别系统就是利用图像分类器将图像分到已知的类别中。 ERF在机器学习领域非常流行，因为ERF
具有较快的速度和比较精确的准确度。我们基于图像的特征构建一组决策树，并通过训练这个森
林 实 现 正 确 决 策 。

2、build_features.py : 利用视觉码本和向量量化创建特征,生成 codebook.pkl、feature_map.pkl

3、trainer.py : 利用极端随机森林训练图像分类器 生成 erf.pkl

4、object_recognizer.py : 对象识别器