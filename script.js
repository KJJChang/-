// 獲取 canvas 和繪圖上下文
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');


// 設置 Canvas 背景顏色為黑色
   function setBackground() {
     ctx.fillStyle = 'black';
     ctx.fillRect(0, 0, canvas.width, canvas.height);
   }

setBackground();

// 設定畫筆參數
ctx.lineWidth = 10;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';


// 設定繪畫狀態
let drawing = false;

// 開始繪畫
canvas.addEventListener('mousedown', () => {
    drawing = true;
    ctx.beginPath();
});

// 當滑鼠移動時繪畫
canvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        ctx.lineTo(event.offsetX, event.offsetY);
        ctx.stroke();
    }
});

// 停止繪畫
canvas.addEventListener('mouseup', () => {
    drawing = false;
});

// 清除畫布
document.getElementById('clearButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setBackground();
});

// 提交繪製的數字圖像
document.getElementById('submitButton').addEventListener('click', () => {
    // 將 canvas 轉換為 base64 編碼圖像
    const dataURL = canvas.toDataURL('image/png');

    // 發送到後端
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })  // 將 base64 圖像數據包裝到 JSON 中
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        alert(`Prediction: ${data.prediction}`);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
