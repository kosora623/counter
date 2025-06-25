// HTML要素の取得
const startCountBtn = document.getElementById('startCountBtn');
const personCountBtn = document.getElementById('personCountBtn');
const endCountBtn = document.getElementById('endCountBtn');
const downloadExcelBtn = document.getElementById('downloadExcelBtn');
const undoLastCountBtn = document.getElementById('undoLastCountBtn'); // 新しいボタンの取得

const currentCountSpan = document.getElementById('currentCount');
const startTimeSpan = document.getElementById('startTime');
const endTimeSpan = document.getElementById('endTime');
const timestampsList = document.getElementById('timestampsList');

// データ構造（ローカルストレージなし）
let currentSession = {
    count: 0,
    timestamps: [], // 各人の出現時刻をここに記録
    startTime: null,
    endTime: null
};

// 初期表示の更新
document.addEventListener('DOMContentLoaded', () => {
    updateDisplay();
    // 初期状態ではカウントボタン、終了ボタン、取り消しボタンは無効
    personCountBtn.disabled = true;
    endCountBtn.disabled = true;
    undoLastCountBtn.disabled = true; // 新しいボタンも初期は無効
});

// 表示を更新する関数
function updateDisplay() {
    currentCountSpan.textContent = currentSession.count;
    startTimeSpan.textContent = currentSession.startTime ? new Date(currentSession.startTime).toLocaleString() : '--';
    endTimeSpan.textContent = currentSession.endTime ? new Date(currentSession.endTime).toLocaleString() : '--';

    timestampsList.innerHTML = ''; // リストをクリア
    currentSession.timestamps.forEach(timestamp => {
        const li = document.createElement('li');
        li.textContent = new Date(timestamp).toLocaleString();
        timestampsList.appendChild(li);
    });

    // ☆ここから追加・変更☆
    // カウント履歴のリストを一番下までスクロールさせる
    timestampsList.scrollTop = timestampsList.scrollHeight;
    // ☆ここまで追加・変更☆

    // カウント履歴がある場合にのみ取り消しボタンを有効化
    undoLastCountBtn.disabled = currentSession.timestamps.length === 0;
}

// 集計開始ボタンのイベントリスナー
startCountBtn.addEventListener('click', () => {
    // 現在のセッションをリセットし、新しい集計を開始
    currentSession = {
        count: 0,
        timestamps: [],
        startTime: new Date().toISOString(), // ISO形式で保存
        endTime: null
    };
    updateDisplay();
    personCountBtn.disabled = false; // カウントボタンを有効化
    endCountBtn.disabled = false;     // 終了ボタンを有効化
    startCountBtn.disabled = true;    // 開始ボタンを無効化
    // undoLastCountBtn.disabled は updateDisplay() で自動的に設定される
});

// 人カウントボタンのイベントリスナー
personCountBtn.addEventListener('click', () => {
    if (!currentSession.startTime) {
        alert('まず「集計開始」ボタンを押してください。');
        return;
    }
    currentSession.count++;
    const now = new Date().toISOString();
    currentSession.timestamps.push(now); // 各人の出現時刻を記録
    updateDisplay();
});

// 集計終了ボタンのイベントリスナー
endCountBtn.addEventListener('click', () => {
    if (!currentSession.startTime) {
        alert('集計が開始されていません。');
        return;
    }
    currentSession.endTime = new Date().toISOString();
    updateDisplay(); // 終了時刻を更新表示

    personCountBtn.disabled = true; // カウントボタンを無効化
    endCountBtn.disabled = true;     // 終了ボタンを無効化
    startCountBtn.disabled = false;  // 開始ボタンを有効化
    undoLastCountBtn.disabled = true; // 集計終了後は取り消しボタンも無効化
    
    alert('集計が終了しました。');
});

// 新しい機能：一つ前のカウントを取り消しボタンのイベントリスナー
undoLastCountBtn.addEventListener('click', () => {
    if (currentSession.timestamps.length > 0) {
        currentSession.timestamps.pop(); // 配列の最後の要素を削除
        currentSession.count--;          // カウントを減らす
        updateDisplay();                 // 表示を更新
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

    // SheetJS用のデータ準備
    // ヘッダー行
    const ws_data = [
        ["No.", "カウント時刻"]
    ];

    // 各カウントイベントを一行として追加
    currentSession.timestamps.forEach((timestamp, index) => {
        ws_data.push([
            index + 1, // No. (1から始まる)
            new Date(timestamp).toLocaleString() // 各カウント時刻
        ]);
    });

    // ワークシートの作成
    const ws = XLSX.utils.aoa_to_sheet(ws_data);

    // ワークブックの作成とワークシートの追加
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "カウント履歴"); // シート名

    // ファイルの書き出し
    // ファイル名に開始時刻を含める
    const startTimeForFileName = currentSession.startTime ? 
                                new Date(currentSession.startTime).toLocaleDateString('ja-JP', {year: 'numeric', month: '2-digit', day: '2-digit'}).replace(/\//g, '') + 
                                new Date(currentSession.startTime).toLocaleTimeString('ja-JP', {hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}).replace(/:/g, '') : 
                                '不明';
    const fileName = `人数カウント_${startTimeForFileName}.xlsx`;
    XLSX.writeFile(wb, fileName);
});