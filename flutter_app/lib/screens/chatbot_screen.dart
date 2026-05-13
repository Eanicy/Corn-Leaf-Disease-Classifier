import 'package:flutter/material.dart';
import '../services/chatbot_service.dart';

class _ChatMessage {
  final String text;
  final bool isUser;
  _ChatMessage({required this.text, required this.isUser});
}

class ChatbotScreen extends StatefulWidget {
  final String? detectedDisease;

  const ChatbotScreen({Key? key, this.detectedDisease}) : super(key: key);

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  final ChatbotService _chatbot = ChatbotService();
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<_ChatMessage> _messages = [];
  bool _isInitialized = false;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initializeChatbot();
  }

  Future<void> _initializeChatbot() async {
    try {
      await _chatbot.initialize();

      // Build welcome message
      final disease = widget.detectedDisease;
      final String welcome;
      if (disease != null && disease.isNotEmpty && disease != 'Healthy') {
        final d = disease.replaceAll('_', ' ');
        welcome =
            'Hi! I detected $d in your corn plant. Ask me anything about its symptoms, treatment, or prevention.';
      } else {
        welcome =
            'Hi! I can help you with questions about corn leaf diseases. What would you like to know?';
      }

      setState(() {
        _messages.add(_ChatMessage(text: welcome, isUser: false));
        _isInitialized = true;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error initializing chatbot: $e')),
        );
      }
    }
  }

  Future<void> _sendMessage(String message) async {
    if (message.trim().isEmpty) return;

    setState(() {
      _messages.add(_ChatMessage(text: message, isUser: true));
      _isLoading = true;
    });
    _messageController.clear();
    _scrollToBottom();

    try {
      final response = await _chatbot.getResponse(
        message,
        diseaseContext: widget.detectedDisease,
      );

      if (mounted) {
        setState(() {
          _messages.add(_ChatMessage(text: response, isUser: false));
          _isLoading = false;
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Disease Chatbot'),
          elevation: 0,
          backgroundColor: Colors.green,
        ),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    final suggestedQuestions =
        _chatbot.getSuggestedQuestions(disease: widget.detectedDisease);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Disease Chatbot'),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.green,
      ),
      body: Column(
        children: [
          // Chat messages
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                return Align(
                  alignment:
                      msg.isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.symmetric(vertical: 6),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 12),
                    constraints: BoxConstraints(
                      maxWidth: MediaQuery.of(context).size.width * 0.75,
                    ),
                    decoration: BoxDecoration(
                      color: msg.isUser ? Colors.green[600] : Colors.grey[200],
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      msg.text,
                      style: TextStyle(
                        color: msg.isUser ? Colors.white : Colors.black87,
                        fontSize: 14,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),

          // Loading indicator
          if (_isLoading)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Row(
                children: [
                  SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor:
                          AlwaysStoppedAnimation<Color>(Colors.green[600]!),
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Text('Thinking…',
                      style: TextStyle(fontSize: 12, color: Colors.grey)),
                ],
              ),
            ),

          // Suggested questions (only when just the welcome message is shown)
          if (_messages.length <= 1)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 4),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Padding(
                    padding: EdgeInsets.only(left: 4, bottom: 6),
                    child: Text('Suggested questions:',
                        style: TextStyle(fontSize: 12, color: Colors.grey)),
                  ),
                  Wrap(
                    spacing: 8,
                    runSpacing: 6,
                    children: suggestedQuestions
                        .map(
                          (q) => GestureDetector(
                            onTap: () => _messageController.text = q,
                            child: Chip(
                              label: Text(q,
                                  style: const TextStyle(fontSize: 12)),
                              backgroundColor: Colors.green[50],
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(20),
                                side:
                                    BorderSide(color: Colors.green[300]!),
                              ),
                            ),
                          ),
                        )
                        .toList(),
                  ),
                ],
              ),
            ),

          // Input area
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white,
              border:
                  Border(top: BorderSide(color: Colors.grey[200]!)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _messageController,
                    enabled: !_isLoading,
                    decoration: InputDecoration(
                      hintText: 'Ask about corn diseases…',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(24),
                        borderSide:
                            BorderSide(color: Colors.grey[300]!),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 12),
                      isDense: true,
                    ),
                    onSubmitted: _isLoading
                        ? null
                        : (text) => _sendMessage(text),
                  ),
                ),
                const SizedBox(width: 8),
                FloatingActionButton(
                  mini: true,
                  onPressed: _isLoading
                      ? null
                      : () => _sendMessage(_messageController.text),
                  backgroundColor: Colors.green,
                  child: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }
}
