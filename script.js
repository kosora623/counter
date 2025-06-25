// HTML要素の取得
const startCountBtn = document.getElementById('startCountBtn');
const personCountBtn = document.getElementById('personCountBtn');
const endCountBtn = document.getElementById('endCountBtn');
const downloadExcelBtn = document.getElementById('downloadExcelBtn');
const undoLastCountBtn = document.getElementById('undoLastCountBtn');

const currentCountSpan = document.getElementById('currentCount');
const startTimeSpan = document.getElementById('startTime');
const endTimeSpan = document.getElementById('endTime');
const timestampsList = document.getElementById('timestampsList');

// データ構造（ローカルストレージなし）
let currentSession = {
    count: 0,
    timestamps: [],
    startTime: null,
    endTime: null
};

// 初期表示の更新
document.addEventListener('DOMContentLoaded', () => {
    updateDisplay();
    personCountBtn.disabled = true;
    endCountBtn.disabled = true;
    undoLastCountBtn.disabled = true;
});

// 表示を更新する関数
function updateDisplay() {
    currentCountSpan.textContent = currentSession.count;
    startTimeSpan.textContent = currentSession.startTime ? new Date(currentSession.startTime).toLocaleString() : '--';
    endTimeSpan.textContent = currentSession.endTime ? new Date(currentSession.endTime).toLocaleString() : '--';

    timestampsList.innerHTML = '';
    currentSession.timestamps.forEach(timestamp => {
        const li = document.createElement('li');
        li.textContent = new Date(timestamp).toLocaleString();
        timestampsList.appendChild(li);
    });

    timestampsList.scrollTop = timestampsList.scrollHeight;

    undoLastCountBtn.disabled = currentSession.timestamps.length === 0;
}

// 集計開始ボタンのイベントリスナー
startCountBtn.addEventListener('click', () => {
    currentSession = {
        count: 0,
        timestamps: [],
        startTime: new Date().toISOString(),
        endTime: null
    };
    updateDisplay();
    personCountBtn.disabled = false;
    endCountBtn.disabled = false;
    startCountBtn.disabled = true;
});

// 人カウントボタンのイベントリスナー
personCountBtn.addEventListener('click', () => {
    if (!currentSession.startTime) {
        alert('まず「集計開始」ボタンを押してください。');
        return;
    }
    currentSession.count++;
    const now = new Date().toISOString();
    currentSession.timestamps.push(now);
    updateDisplay();
});

// 集計終了ボタンのイベントリスナー
endCountBtn.addEventListener('click', () => {
    if (!currentSession.startTime) {
        alert('集計が開始されていません。');
        return;
    }
    currentSession.endTime = new Date().toISOString();
    updateDisplay();

    personCountBtn.disabled = true;
    endCountBtn.disabled = true;
    startCountBtn.disabled = false;
    undoLastCountBtn.disabled = true;
    
    alert('集計が終了しました。');
});

// 一つ前のカウントを取り消しボタンのイベントリスナー
undoLastCountBtn.addEventListener('click', () => {
    if (currentSession.timestamps.length > 0) {
        currentSession.timestamps.pop();
        currentSession.count--;
        updateDisplay();
    } else {
        alert('取り消せるカウントがありません。');
    }
});

// Excelダウンロードボタンのイベントリスナー
downloadExcelBtn.addEventListener('click', () => {
    if (currentSession.timestamps.length === 0) {
        alert('ダウンロードするデータがありません。');
        return;
    }

    const ws_data = [
        ["No.", "カウント時刻"]
    ];

    currentSession.timestamps.forEach((timestamp, index) => {
        ws_data.push([
            index + 1,
            new Date(timestamp).toLocaleString()
        ]);
    });

    const ws = XLSX.utils.aoa_to_sheet(ws_data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "カウント履歴");

    const startTimeForFileName = currentSession.startTime ? 
                                new Date(currentSession.startTime).toLocaleDateString('ja-JP', {year: 'numeric', month: '2-digit', day: '2-digit'}).replace(/\//g, '') + 
                                new Date(currentSession.startTime).toLocaleTimeString('ja-JP', {hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}).replace(/:/g, '') : 
                                '不明';
    const fileName = `人数カウント_${startTimeForFileName}.xlsx`;
    XLSX.writeFile(wb, fileName);
});