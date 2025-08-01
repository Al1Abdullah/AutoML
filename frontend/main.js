document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const loader = document.querySelector('.loader');

    function animateNavText() {
        const navTexts = document.querySelectorAll('.nav-text');
        navTexts.forEach(text => {
            text.style.animation = 'none';
            text.offsetHeight; // Trigger reflow
            text.style.animation = '';
        });
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageId = link.dataset.page;

            pages.forEach(page => {
                page.classList.remove('active');
            });

            navLinks.forEach(navLink => {
                navLink.classList.remove('active');
            });

            document.getElementById(pageId).classList.add('active');
            link.classList.add('active');
        });
    });

    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        mainContent.classList.toggle('collapsed');
        if (!sidebar.classList.contains('collapsed')) {
            animateNavText();
        }
    });

    function formatAIResponse(text) {
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/^\d+\.\s+(.*)/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/s, '<ol>$1<\/ol>');
        text = text.replace(/^\*\s+(.*)/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1<\/ul>');
        return text;
    }

    const csvUpload = document.getElementById('csv-upload');
    const uploadStatus = document.getElementById('upload-status');
    const columnList = document.getElementById('column-list');
    const plotType = document.getElementById('plot-type');
    const plotCol1 = document.getElementById('plot-col1');
    const plotCol2 = document.getElementById('plot-col2');
    const scatterColorContainer = document.getElementById('scatter-color-container');
    const scatterColor = document.getElementById('scatter-color');
    const generatePlot = document.getElementById('generate-plot');
    const plotImg = document.getElementById('plot-img');
    const plotError = document.getElementById('plot-error');
    const learningType = document.getElementById('learning-type');
    const modelDropdown = document.getElementById('model-dropdown');
    const targetColumnDropdown = document.getElementById('target-column-dropdown');
    const trainModel = document.getElementById('train-model');
    const trainOutput = document.getElementById('train-output');
    const aiQuestion = document.getElementById('ai-question');
    const askAi = document.getElementById('ask-ai');
    const aiAnswer = document.getElementById('ai-answer');

    csvUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        uploadStatus.textContent = 'Uploading...';
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            uploadStatus.textContent = result.message || result.error;
            if (response.ok) {
                updateColumnSelectors();
                setLearningType();
                updatePlotOptions();
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
        }
    });

    async function updateColumnSelectors() {
        try {
            const response = await fetch('/api/columns');
            const result = await response.json();
            const columns = result.columns || [];

            [columnList, plotCol1, plotCol2, targetColumnDropdown, scatterColor].forEach(selector => {
                selector.innerHTML = '';
                const defaultOption = document.createElement('option');
                defaultOption.value = 'None';
                defaultOption.textContent = 'None';
                selector.appendChild(defaultOption);

                columns.forEach(col => {
                    const option = document.createElement('option');
                    option.value = col;
                    option.textContent = col;
                    selector.appendChild(option);
                });
            });
        } catch (error) {
            console.error('Error updating column selectors:', error);
        }
    }

    plotType.addEventListener('change', () => {
        if (plotType.value === 'Scatter') {
            scatterColorContainer.style.display = 'block';
        } else {
            scatterColorContainer.style.display = 'none';
        }
    });

    generatePlot.addEventListener('click', async () => {
        if (!plotType.value) {
            plotError.textContent = 'Please select a plot type.';
            return;
        }

        loader.style.display = 'block';
        plotImg.src = '';
        plotError.textContent = '';

        const body = {
            plot_type: plotType.value,
            col1: plotCol1.value,
            col2: plotCol2.value,
            color_col: scatterColor.value
        };

        try {
            const response = await fetch('/api/plot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const result = await response.json();
            if (result.image) {
                plotImg.src = `data:image/png;base64,${result.image}`;
            } else {
                plotError.textContent = result.error;
            }
        } catch (error) {
            plotError.textContent = `Error: ${error.message}`;
        } finally {
            loader.style.display = 'none';
        }
    });

    function formatMetrics(metrics) {
        let formatted = '\n';
        for (const [key, value] of Object.entries(metrics)) {
            formatted += `<strong>${key}:</strong> ${JSON.stringify(value, null, 2)}\n`;
        }
        return formatted;
    }

    async function setLearningType() {
        try {
            const response = await fetch('/api/learning_type');
            const result = await response.json();
            if (result.learning_type) {
                learningType.disabled = false;
                learningType.value = result.learning_type;
                learningType.dispatchEvent(new Event('change'));
                learningType.disabled = true;
                if (result.learning_type === 'Supervised' && result.target_column) {
                    targetColumnDropdown.value = result.target_column;
                }
            }
        } catch (error) {
            console.error('Error setting learning type:', error);
        }
    }

    async function updatePlotOptions() {
        try {
            const response = await fetch('/api/plot_options');
            const result = await response.json();
            const plots = result.plots || [];

            plotType.innerHTML = '';
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Select Plot Type';
            plotType.appendChild(defaultOption);

            plots.forEach(plotName => {
                const option = document.createElement('option');
                option.value = plotName;
                option.textContent = plotName;
                plotType.appendChild(option);
            });
        } catch (error) {
            console.error('Error updating plot options:', error);
        }
    }

    trainModel.addEventListener('click', async () => {
        trainOutput.textContent = 'Training in progress...';
        loader.style.display = 'block';
        const body = {
            learning_type: learningType.value,
            model_name: modelDropdown.value,
            target_col: targetColumnDropdown.value
        };

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const result = await response.json();
            let output = result.message || result.error;
            if (result.metrics) {
                output += formatMetrics(result.metrics);
            }
            if (result.result) {
                output += `\n<strong>Result:</strong> ${JSON.stringify(result.result, null, 2)}`;
            }
            trainOutput.innerHTML = output;
        } catch (error) {
            trainOutput.textContent = `Error: ${error.message}`;
        } finally {
            loader.style.display = 'none';
        }
    });

    askAi.addEventListener('click', async () => {
        aiAnswer.textContent = 'Thinking...';
        loader.style.display = 'block';
        const body = {
            user_query: aiQuestion.value
        };

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const result = await response.json();
            aiAnswer.innerHTML = formatAIResponse(result.answer || result.error);
        } catch (error) {
            aiAnswer.textContent = `Error: ${error.message}`;
        } finally {
            loader.style.display = 'none';
        }
    });

    learningType.addEventListener('change', () => {
        const supervisedModels = ["Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest", "SVM", "KNN", "XGBoost", "CatBoost", "Linear Regression"];
        const unsupervisedModels = ["KMeans", "DBSCAN", "PCA"];
        const models = learningType.value === 'Supervised' ? supervisedModels : unsupervisedModels;
        
        modelDropdown.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelDropdown.appendChild(option);
        });
    });

    learningType.dispatchEvent(new Event('change'));

    // Add click-to-copy functionality to output boxes
    [trainOutput, aiAnswer, uploadStatus, plotError].forEach(el => {
        el.addEventListener('click', () => {
            const textToCopy = el.textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalText = el.textContent;
                el.textContent = 'Copied!';
                setTimeout(() => {
                    el.textContent = originalText;
                }, 1000);
            });
        });
    });
});
