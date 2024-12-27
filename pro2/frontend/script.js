document.getElementById('summarizeButton').addEventListener('click', async function() {
    const report = document.getElementById('reportInput').value;
    
    if (!report) {
        alert("Please enter a medical report to summarize.");
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:5000/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ report })  
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('summaryOutput').textContent = result.summary;
        } else {
            const error = await response.json();
            alert("Error: " + error.error);
        }
    } catch (err) {
        console.error("Error during fetch operation:", err);
        alert("An error occurred while summarizing the report. Please try again later.");
    }
});
