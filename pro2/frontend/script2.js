document.getElementById('uploadButton').addEventListener('click', async function() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an Excel file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:5000/summarize', {
            method: 'POST',
            body: formData
            
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'summarized_reports.xlsx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.getElementById('statusMessage').textContent = "Summarization complete! Downloading file...";
        } else {
            const errorData = await response.json();
            alert("Error: " + errorData.error);
        }
    } catch (err) {
        console.error("Error during fetch operation:", err);
        alert("An error occurred while summarizing the reports.");
    }
});
