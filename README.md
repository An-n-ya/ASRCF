# 中期检查

## 主要任务

  （1）熟悉数字图像处理相关理论知识；

  （2）学习matlab编程；

  （3）学习核相关滤波相关理论知识；

  （4）实现基于自适应空间加权相关滤波的视觉目标跟踪算法，并进行实验验证。

  （5）严格按照中国石油大学(华东)本科毕业设计(论文)环节管理规定和“控制科学与工程学院2020届毕业设计（论文）相关规定”撰写论文。

## 任务目标

* 深入了解课题研究的目的、内容，整理研究思路、具体需要解决的问题和预期目标等
* 深入学习、研究相关滤波算法，实现基本的相关滤波方法，并总结基本相关滤波方法的优缺点
* 实现基于自适应的相关滤波算法
* 基于自适应空间加权相关滤波，并进行跟踪实验验证

## 项目工作进展

### 已完成任务

- [x] 相关文献的阅读、翻译，开题报告及文献综述的撰写
- [x] 理解了相关滤波算法的实现原理、优缺点
- [x] 实现了最原始的相关滤波算法MOSSE
- [x] 实现了自适应相关滤波算法ASRCF
- [x] 基于OTB平台对各种相关滤波算法进行了评估
- [x] 针对ASRCF算法的优化

### 相关滤波理论

* 相关性

  信号处理中的相关性，指的是两个元素之间的联系。按元素的类别，可以分为互相关(两个信号之间的联系)和自相关(信号与自身的相关性)。假设有两个信号f和g，则两个信号的相关性为：

![image-20200418113210909](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418113210909.png)

​	举个例子，现有如下仓鼠

​	![image-20200418113244512](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418113244512.png)

​	将上面两幅图做相关运算，得到结果如下图：

![image-20200418113335447](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418113335447.png)

注意到，图像的中心，也就是原图中仓鼠所在的位置，产生了波峰，说明追踪目标与原图中心有较好的相关度。
利用这样的性质，便可以考虑设计相关滤波器。

### 相关滤波的优缺点

* 优点

  基于相关滤波理论的跟踪方案在频域内进行计算有效控制了运算成本,提高了跟踪效率.

* 缺点

  #### 边界效应

  MOSSE的种种问题成为了后来相关滤波器的研究方向，其中，关于相关滤波器最大的问题便是“边界效应”。

  由于相关滤波器引入了频域操作，而二维相关操作需要进行周期延拓，所以实际的进入相关操作的样本如下图。

  ![image-20200418114933846](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418114933846.png)

  当目标移动到边缘时，就可能会形成“错影”，如下图，导致跟踪器跟错目标。

  ![image-20200418115125260](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418115125260.png)

### MOSSE相关滤波实现

MOSSE模型如下：

​	![image-20200418113738153](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418113738153.png)

其中 \hat h代表h的傅里叶变换，y为高斯分布，x为输入信号，h为滤波器。在MOSSE中，所有的操作都是元素级别的，因此使用的是Frobenius范数。

对于上述优化问题，直接求导便可以得到最优解为：

![image-20200418113826819](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418113826819.png)

但这样得到的滤波器仅对某一帧有效，为了使滤波器对跟踪目标的形变、光照等外界影响有更好的鲁棒性，采取了如下更新策略：

![image-20200418113845367](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418113845367.png)

使用matlab编写程序：

单目标跟踪的目标确定于第一帧，因此模型对于第一帧的学习至关重要，于是程序应该对第一帧进行着重学习：

下面是对第一帧中目标的学习：

```matlab
g = gaussC(R,C, sigma, center);%通过R和C产生size等同于im的高斯滤波函数
g = mat2gray(g);
img = imcrop(img, rect);
g = imcrop(g, rect);%将高斯函数的大小裁剪为目标大小
G = fft2(g);%将高斯滤波函数变换到频域
Ai = (G.*conj(fft2(fi)));%计算上式中的ai
Bi = (fft2(fi).*conj(fft2(fi)));%计算上式中的bi
N = 128;%确定循环次数

for i = 1:N%将第一帧目标随机旋转N次，让模型对其学习N次 
    fi = preprocess(rand_warp(img));%旋转目标
    Ai = Ai + (G.*conj(fft2(fi)));%更新ai
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));%更新bi
end
```

学习完第一帧，之后便是线上学习了。

```matlab
Hi = Ai./Bi;%由ai,bi得出滤波器
fi = imcrop(img, rect);%根据目标大小裁剪输入图像            
gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi))));%滤波器与图像在频域进行点积，再进行逆变换
maxval = max(gi(:))
[P, Q] = find(gi == maxval);%找出最大值的坐标
dx = mean(P)-height/2;
dy = mean(Q)-width/2;%算出偏移量

rect = [rect(1)+dy rect(2)+dx width height];%根据跟踪结果计算出新的选框
fi = imcrop(img, rect); %根据预测选框对当前帧进行裁剪
Ai = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai;%更新ai
Bi = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi;%更新bi
```

MOSSE的跟踪效果如下：

![MOSSE_result](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/MOSSE_result.gif)

可见，最原始的相关滤波器不能很好的应对目标的形变、尺度变化。

## 空间自适应权重相关滤波的推导以及实现

ASRCF针对相关滤波算法存在的边界效应提出了自适应的惩罚项，能够对目标进行针对性的学习。

* 推导

ASRCF不把w看做常数，而看成一个待优化的量，并在后面再添加一项带有先验信息的惩罚项。

![image-20200418120748490](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418120748490.png)

对于该目标方程的优化参见https://github.com/ankh04/ASRCF/derivation.pdf

对于ADMM的推导参见https://github.com/ankh04/ASRCF/ADMM.pdf

* 实现

具体程序很复杂，这里展示比较重要的。

1. 特征提取

   ```matlab
   x = vl_simplenn(vggmnet,img);%通过matlab内置函数提取特征
   ```

   使用了三种特征：VGG16，VGG-M，fHOG

   ```matlab
   x_vgg16= get_vggfeatures(patch,use_sz,23);%提取VGG16特征
   x_vggm=get_vggmfeatures(patch,use_sz,4);%提取VGG16特征
   x_hc=get_features(patch,hogfeatures,hogparams);%提取VGG16特征
   featuremap={x_hc*pe(1),x_vgg16*pe(2),x_vggm*pe(3)};%将三种特征按照一定的比例进行合并
   ```

2. fDSST

   预先定义好五种缩放因子，跟踪五个缩放因子进行采样

   ```matlab
   for scale_ind = 1:nScales        
       multires_pixel_template(:,:,:,scale_ind) = ...
       get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);       
   end
   ```

   得出对由五个不同尺寸组成的样本金字塔得到的响应图后，找到响应和最大的。

   ```matlab
   max_response = 1 / prod(use_sz) * real(mtimesx(mtimesx(exp_iky, responsef, 'speed'), exp_ikx, 'speed'));
   ```

   这样便确定了最佳的缩放因子。

3. 空间自适应优化

   对w采用ADMM算法求解，过程如下：

   ![image-20200418170244584](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418170244584.png)

   ```matlab
   while (i <= params.admm_iterations)
       w = bsxfun(@rdivide,(q-m),(1+(params.admm_lambda1/mu)*Hh));
       q=(params.admm_lambda2*model_w + mu*(w+m))/(params.admm_lambda2 + mu);
       m = m + (w - q);
       mu = min(betha * mu, mumax);
       i = i+1;       
   end
   ```

## 在OTB平台上对算法进行评估

评估结果如下：

![image-20200418171457399](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418171457399.png)

DiMP与ATOM都是VOT2019中的SOTA跟踪器。GradNet是与ASRCF同出自大连理工大学的卢湖川团
队，这两个跟踪器都上了CVPR2019。

OTB对目标跟踪领域难题划分了11种难题，ASRCF在其中的两种表现得较差：

在尺度缩放问题下：

![image-20200418171610597](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418171610597.png)

低分辨率下：

![image-20200418171736732](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418171736732.png)

## 对空间自适应权重滤波算法的优化

* FFT算法加速

  由于输入的图像均是实数矩阵，而实数矩阵的傅里叶变换是共轭对称的，所以在频域计算时可以只计算一半，这样便可以把计算量降低至50%。

* 模型更新策略

  采用了APCE（average peak-to correlation energy）这以置信度指标，只有在跟踪置信度比较高的时候才更新跟踪模型，避免目标模型被污染。APCE反应反应响应图的波动程度和检测目标的置信水平

  ![image-20200418172117374](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418172117374.png)

  g为response map

  将APCE融入特别的模型更新函数，使得跟踪器能够对目标进行时间尺度上的自适应更新。

  ![image-20200418172512622](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418172512622.png)

  下图为otb benchmark下的skating2评分

  ![image-20200418172539239](https://github.com/ankh04/ASRCF/blob/master/fig/中期检查/image-20200418172539239.png)