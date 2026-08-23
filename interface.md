
POST /model/ocr

功能: 获取图片中的文本

请求参数:

```json
{
    image: base64图片,
    threshold: 0.3, // 可选参数
    box_threshold: 0.6 // 可选参数
}
```

返回结果:

```json
{
    code: 0,
    msg: "success",
    result: [
        [
            [[point0_x, point0_y],[point1_x, point1_y],[point2_x, point2_y],[point3_x, point3_y]],["text", score]
        ],
        ...
    ]
}
{
    code: 1,
    msg: "unknow error",
    result: []
}
```