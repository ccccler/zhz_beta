document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const caseAnalysis = document.getElementById('caseAnalysis');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');

    // 添加聊天页面的欢迎消息
    addMessage("你好！我是张洪忠。让我们开始谈论智能传播的问题吧！\n本智能体数据截至本人最新研究，请注意回答时效性。", 'bot', chatMessages);

    // 添加案例分析页面的欢迎消息
    addMessage("你好！我是张洪忠。此对话框可结合本人既有研究，对其他领域问题做延展讨论，请在这里输入你想要分析的案例。", 'bot', caseAnalysis);

    // 标签页切换功能
    const tabButtons = document.querySelectorAll('.tab-button');
    const chatContents = document.querySelectorAll('.chat-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // 移除所有活动状态
            tabButtons.forEach(btn => btn.classList.remove('active'));
            chatContents.forEach(content => content.classList.remove('active'));

            // 添加当前选中的活动状态
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            const activeContent = tabId === 'chat' ? 
                document.getElementById('chatMessages') : 
                document.getElementById('caseAnalysis');
            activeContent.classList.add('active');
        });
    });

    // 发送消息功能
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (message) {
            const activeTab = document.querySelector('.tab-button.active').getAttribute('data-tab');
            const targetContainer = activeTab === 'chat' ? chatMessages : caseAnalysis;
            
            addMessage(message, 'user', targetContainer);
            messageInput.value = '';

            try {
                console.log('Sending message:', message);
                
                // 添加加载消息
                const loadingMessage = addLoadingMessage(targetContainer);
                
                // 根据不同的标签页使用不同的API端点
                const apiEndpoint = activeTab === 'chat' ? '/api/query' : '/api/case-analysis';
                const eventSource = new EventSource(`${apiEndpoint}?question=${encodeURIComponent(message)}`);
                
                let isFirstMessage = true;
                
                eventSource.onmessage = function(event) {
                    if (isFirstMessage) {
                        // 收到第一条消息时，移除加载动画
                        removeLoadingMessage(loadingMessage);
                        
                        // 创建新的消息容器
                        const messageDiv = document.createElement('div');
                        messageDiv.classList.add('message', 'bot-message');
                        
                        const avatarImg = document.createElement('img');
                        avatarImg.src = '../static/assets/bot-avatar.jpg';
                        avatarImg.alt = '机器人头像';
                        avatarImg.classList.add('message-avatar');
                        
                        const messageContent = document.createElement('div');
                        messageContent.classList.add('message-content');
                        messageContent.innerText = event.data;
                        
                        messageDiv.appendChild(avatarImg);
                        messageDiv.appendChild(messageContent);
                        targetContainer.appendChild(messageDiv);
                        
                        isFirstMessage = false;
                    } else {
                        // 后续消息直接追加到现有消息内容中
                        const lastMessage = targetContainer.lastElementChild;
                        const messageContent = lastMessage.querySelector('.message-content');
                        messageContent.innerText += event.data;
                    }
                    targetContainer.scrollTop = targetContainer.scrollHeight;
                };
                
                eventSource.onerror = function(error) {
                    console.error('EventSource error:', error);
                    eventSource.close();
                    removeLoadingMessage(loadingMessage);
                    if (!document.querySelector('.bot-message:not(.loading-message)')) {
                        addMessage('抱歉，发生了错误。', 'bot', targetContainer);
                    }
                };
                
                eventSource.addEventListener('end', function(event) {
                    eventSource.close();
                    removeLoadingMessage(loadingMessage);
                });

            } catch (error) {
                console.error('Error details:', error);
                addMessage(`抱歉，发生了错误：${error.message}`, 'bot', targetContainer);
            }
        }
    }

    // 添加消息到聊天界面
    function addMessage(text, type, targetContainer) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${type}-message`);

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.innerText = text;

        if (type === 'user') {
            messageDiv.appendChild(messageContent);
        } else {
            const avatarImg = document.createElement('img');
            avatarImg.onerror = function() {
                console.error(`头像加载失败，请检查图片路径: ${this.src}`);
            };
            avatarImg.src = '../static/assets/bot-avatar.jpg';
            avatarImg.alt = '机器人头像';
            avatarImg.classList.add('message-avatar');
            
            messageDiv.appendChild(avatarImg);
            messageDiv.appendChild(messageContent);
        }
        
        targetContainer.appendChild(messageDiv);
        targetContainer.scrollTop = targetContainer.scrollHeight;
    }

    // 事件监听器
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 添加这些新函数
    function addLoadingMessage(targetContainer) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'bot-message', 'loading-message');
        
        const avatarImg = document.createElement('img');
        avatarImg.src = '../static/assets/bot-avatar.jpg';
        avatarImg.alt = '机器人头像';
        avatarImg.classList.add('message-avatar');
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        // 添加加载文本和进度条容器
        const loadingText = document.createElement('div');
        loadingText.classList.add('loading-text');
        loadingText.textContent = '数据检索中';
        
        const progressBar = document.createElement('div');
        progressBar.classList.add('progress-bar');
        const progressFill = document.createElement('div');
        progressFill.classList.add('progress-fill');
        progressBar.appendChild(progressFill);
        
        messageContent.appendChild(loadingText);
        messageContent.appendChild(progressBar);
        
        messageDiv.appendChild(avatarImg);
        messageDiv.appendChild(messageContent);
        
        targetContainer.appendChild(messageDiv);
        targetContainer.scrollTop = targetContainer.scrollHeight;
        
        // 启动进度条动画
        startProgressAnimation(progressFill);
        
        return messageDiv;
    }

    // 修改进度条动画函数
    function startProgressAnimation(progressFill) {
        let progress = 0;
        const interval = setInterval(() => {
            progress += 0.5;  // 将每次增加的进度从1改为0.5
            if (progress <= 100) {
                progressFill.style.width = `${progress}%`;
            } else {
                clearInterval(interval);
            }
        }, 100);  // 将间隔时间从50ms改为100ms

        // 保存interval ID到元素上，以便之后清除
        progressFill.dataset.intervalId = interval;
    }

    // 修改 removeLoadingMessage 函数
    function removeLoadingMessage(loadingMessageDiv) {
        if (loadingMessageDiv && loadingMessageDiv.parentNode) {
            // 清除进度条动画
            const progressFill = loadingMessageDiv.querySelector('.progress-fill');
            if (progressFill && progressFill.dataset.intervalId) {
                clearInterval(parseInt(progressFill.dataset.intervalId));
            }
            loadingMessageDiv.remove();
        }
    }

    // 添加刷新按钮事件监听
    const refreshButton = document.getElementById('refreshPrompts');
    if (refreshButton) {
        refreshButton.addEventListener('click', refreshPromptCards);
    }
});