# Bread-ONNX-Sample
Low-Light Image Enhancementモデルである[Bread](https://github.com/mingcv/bread)のPythonでのONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。変換自体を試したい方は[Convert2ONNX.ipynb](Convert2ONNX.ipynb)を使用ください。<br>

https://user-images.githubusercontent.com/37477845/215316878-dbd191c0-726c-46a2-9fff-55048b9e58a6.mp4

# Requirement 
* OpenCV 4.5.3.56 or later
* onnxruntime 1.12.0 or later
* Pytorch 1.8 or later ※ONNX変換を実施する場合のみ

# Demo
デモの実行方法は以下です。
```bash
python demo_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/Bread_320x240.onnx

# Reference
* [mingcv/Bread](https://github.com/mingcv/bread)
* [Dovyski/cvui](https://github.com/Dovyski/cvui)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Bread-ONNX-Sample is under [Apache2.0 License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[雨イメージ　夜の道路を走る車](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002161702_00000)を使用しています。
