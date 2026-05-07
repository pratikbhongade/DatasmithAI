document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const textInput = document.getElementById('text-input');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const chatContainer = document.getElementById('chat-container');
    const sendBtn = document.getElementById('send-btn');
    const costDisplay = document.getElementById('cost-display');
    const template = document.getElementById('message-template');

    let currentFile = null;
    let totalCost = 0;
    
    // Generate unique session ID
    const sessionId = 'session-' + Date.now() + '-' + Math.floor(Math.random() * 1000);
    
    // Setup WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/${sessionId}`;
    let ws;

    function setupWs() {
        ws = new WebSocket(wsUrl);
        ws.onclose = () => {
            console.log("WebSocket disconnected. Reconnecting in 3s...");
            setTimeout(setupWs, 3000);
        };
    }
    setupWs();

    // Auto-resize textarea
    textInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim() || currentFile) {
            sendBtn.disabled = false;
        } else {
            sendBtn.disabled = true;
        }
    });

    // Handle Enter to submit (Shift+Enter for new line)
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim() || currentFile) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // File Input Logic
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            currentFile = this.files[0];
            fileName.textContent = currentFile.name;
            filePreview.classList.remove('hidden');
            filePreview.classList.add('flex');
            sendBtn.disabled = false;
        }
    });

    removeFileBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        filePreview.classList.add('hidden');
        filePreview.classList.remove('flex');
        if (!textInput.value.trim()) {
            sendBtn.disabled = true;
        }
    });

    // Handle Form Submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const promptText = textInput.value.trim();
        if (!promptText && !currentFile) return;

        // Add user message to chat
        let userMsg = promptText;
        if (currentFile) {
            userMsg = `[Attached: ${currentFile.name}]\n` + userMsg;
        }
        appendMessage('user', userMsg);

        // Reset inputs
        textInput.value = '';
        textInput.style.height = 'auto';
        const fileToSend = currentFile; // Store before clearing
        currentFile = null;
        fileInput.value = '';
        filePreview.classList.add('hidden');
        filePreview.classList.remove('flex');
        sendBtn.disabled = true;

        // Show typing indicator
        const typingId = appendTypingIndicator();

        // Prepare FormData
        const formData = new FormData();
        formData.append('session_id', sessionId);
        if (promptText) formData.append('text_prompt', promptText);
        if (fileToSend) formData.append('file', fileToSend);

        try {
            const response = await fetch('/api/v1/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                removeTypingIndicator(typingId);
                const data = await response.json();
                appendMessage('ai', `Error: ${data.detail || 'Something went wrong.'}`);
                return;
            }

            // We do not await response.json() for the final output here.
            // We just wait for WebSocket messages to populate the message.
            const bubbleId = 'ai-msg-' + Date.now();
            removeTypingIndicator(typingId);
            const msgObj = appendStreamingMessage('ai', bubbleId);

            ws.onmessage = (event) => {
                const msgData = JSON.parse(event.data);
                console.log("msg rx:", msgData);
                
                if (msgData.type === 'log') {
                    addLogToStreamingMessage(msgObj, msgData.step_name, msgData.details);
                } else if (msgData.type === 'token') {
                    appendTokenToStreamingMessage(msgObj, msgData.token);
                } else if (msgData.type === 'result') {
                    console.log("res text:", msgData.extracted_text);
                    if (msgData.cost) {
                        totalCost += msgData.cost;
                        costDisplay.textContent = `Total Cost: $${totalCost.toFixed(4)}`;
                        addCostToStreamingMessage(msgObj, msgData.cost);
                    }
                    if (msgData.extracted_text) {
                        addExtractedTextToMessage(msgObj, msgData.extracted_text);
                    }
                    if (msgData.needs_clarification && msgData.clarification_question) {
                        appendTokenToStreamingMessage(msgObj, `\n\n**Question:** ${msgData.clarification_question}`);
                    }
                } else if (msgData.type === 'error') {
                    appendTokenToStreamingMessage(msgObj, `\n\n**Error:** ${msgData.details}`);
                }
            };

        } catch (error) {
            removeTypingIndicator(typingId);
            appendMessage('ai', `Connection Error: ${error.message}`);
        }
    });

    function appendMessage(role, text, logs = null, cost = null) {
        const clone = template.content.cloneNode(true);
        const container = clone.querySelector('.message-container');
        const bubble = clone.querySelector('.message-bubble');
        
        container.classList.add(role === 'user' ? 'user-message' : 'ai-message');
        
        if (role === 'ai') {
            bubble.innerHTML = marked.parse(text);
            bubble.classList.add('prose', 'prose-sm', 'max-w-none', 'prose-indigo');
            bubble.classList.remove('whitespace-pre-wrap');
        } else {
            bubble.textContent = text;
        }

        if (role === 'ai' && logs && logs.length > 0) {
            const details = clone.querySelector('.logs-details');
            const logsContent = clone.querySelector('.logs-content');
            
            details.classList.remove('hidden');
            
            let logsHtml = '';
            logs.forEach(step => {
                logsHtml += `<div><span class="font-bold text-indigo-400">[${step.step_name}]</span> ${step.details}</div>`;
            });
            if (cost !== null) {
                logsHtml += `<div class="mt-2 pt-2 border-t border-gray-300">Cost for this request: $${cost.toFixed(6)}</div>`;
            }
            logsContent.innerHTML = logsHtml;
        }

        chatContainer.appendChild(clone);
        scrollToBottom();
    }

    // New Streaming Support
    function appendStreamingMessage(role, id) {
        const clone = template.content.cloneNode(true);
        const container = clone.querySelector('.message-container');
        const bubble = clone.querySelector('.message-bubble');
        const details = clone.querySelector('.logs-details');
        const logsContent = clone.querySelector('.logs-content');
        const extractedDetails = clone.querySelector('.extracted-details');
        const extractedContent = clone.querySelector('.extracted-content');
        
        container.id = id;
        container.classList.add('ai-message');
        
        bubble.classList.add('prose', 'prose-sm', 'max-w-none', 'prose-indigo');
        bubble.classList.remove('whitespace-pre-wrap');
        
        // Expose a raw buffer so marked.js can parse markdown cleanly across chunks
        const msgObj = {
            id: id,
            bubble: bubble,
            rawText: '',
            details: details,
            logsContent: logsContent,
            extractedDetails: extractedDetails,
            extractedContent: extractedContent
        };
        
        chatContainer.appendChild(clone);
        scrollToBottom();
        return msgObj;
    }

    function appendTokenToStreamingMessage(msgObj, token) {
        msgObj.rawText += token;
        msgObj.bubble.innerHTML = marked.parse(msgObj.rawText);
        scrollToBottom();
    }

    function addLogToStreamingMessage(msgObj, stepName, detailsStr) {
        msgObj.details.classList.remove('hidden');
        const logHtml = `<div><span class="font-bold text-indigo-400">[${stepName}]</span> ${detailsStr}</div>`;
        msgObj.logsContent.innerHTML += logHtml;
        scrollToBottom();
    }

    function addCostToStreamingMessage(msgObj, cost) {
        msgObj.details.classList.remove('hidden');
        const costHtml = `<div class="mt-2 pt-2 border-t border-gray-300">Cost for this request: $${cost.toFixed(6)}</div>`;
        msgObj.logsContent.innerHTML += costHtml;
        scrollToBottom();
    }

    function addExtractedTextToMessage(msgObj, text) {
        if (!msgObj.extractedDetails) return;
        msgObj.extractedDetails.classList.remove('hidden');
        msgObj.extractedContent.textContent = text;
        scrollToBottom();
    }

    function appendTypingIndicator() {
        const id = 'typing-' + Date.now();
        const clone = template.content.cloneNode(true);
        const container = clone.querySelector('.message-container');
        const bubble = clone.querySelector('.message-bubble');
        
        container.id = id;
        container.classList.add('ai-message');
        
        bubble.innerHTML = `
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        
        chatContainer.appendChild(clone);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
