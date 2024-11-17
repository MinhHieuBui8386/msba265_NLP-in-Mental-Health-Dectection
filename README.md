# NLP in Mental Health Detection: Analysis and Classification
## MSBA 265: Special Analytics Topics
## November 3, 2024
Chun-Yen Lin, Hsin Yu Liao, Minh Hieu Bui, Nehal Bhachu, 
Oprah Winfrey Ewurafua Ampah,  Roopa Sree Gopalappa Seenappa, Suraj Shaik, Thao Nguyen
Dr. Shyla C. Solis
University of The Pacific


1.	Overview
1.1. Purposes
Gen Z faces a significant rise in mental health issues, with studies showing higher levels of anxiety, depression, and stress than in previous generations. For instance, a report from the American Psychological Association (APA) indicates that over 90% of Gen Z adults experienced at least one stress-related physical or emotional symptom in the past year. Additionally, approximately 45% of Gen Z respondents report having a mental health diagnosis, most commonly anxiety and depression, and nearly 60% of college-aged Gen Zers have sought mental health services. Alarmingly, rates of self-harm and suicidal thoughts are also high, with 1 in 4 high school students seriously considering suicide recently. The unique pressures contributing to these challenges include the pervasive influence of social media, which fosters comparison, cyberbullying, and unrealistic standards of beauty, harming self-esteem and mental health. Global crises, such as climate change, economic instability, and the COVID-19 pandemic, also contribute to heightened anxiety, while academic and professional pressures, coupled with economic worries like student debt, add further stress. However, as Gen Z is more open about mental health, they are increasingly aware of and able to identify their struggles. Detecting these issues with NLP can be particularly impactful for Gen Z, as it enables real-time, non-intrusive monitoring of their digital communications—texts, social media, and forums—allowing for passive mental health analysis without disrupting their routines. NLP tools provide scalable support, addressing the challenge of limited mental health service capacity by analyzing language at scale for real-time risk insights. These tools also aid in early detection, allowing intervention before symptoms escalate, while personalization in NLP can adapt models to the specific language patterns and expressions common among Gen Z, including sarcasm and colloquial language, to identify signs of distress with greater accuracy.

1.2.	Users
-	Gen Z and Young Millennials
    - Why Focus on This Group: Gen Z and young millennials face high rates of mental health challenges, including anxiety, depression, and stress. Studies show that social media and digital communication, integral to their lives, often reveal early signs of mental health issues, making this group particularly well-suited for NLP-based analysis.
    - Characteristics: As digital natives, they communicate extensively online, using platforms where text-based analysis is effective. They are generally open to using digital tools for health and wellness, including mental health support.
    - Motivation: Many young people may feel uncomfortable with traditional mental health services due to stigma, cost, or accessibility issues, so a discreet, non-intrusive tool is attractive. This group values privacy and autonomy, making NLP-based tools that analyze language without requiring face-to-face interaction especially appealing.
-	Mental Health Providers and Counselors
    - Why Focus on This Group: Mental health professionals can benefit significantly from insights provided by NLP tools. NLP models can enhance clinicians’ understanding of patients’ mental states, helping detect early signs of deterioration between sessions or adding valuable supplementary data for more personalized care.
    - Characteristics: Mental health providers typically seek data-driven tools that can support their clinical decision-making. NLP tools can help professionals track language patterns or emotional indicators over time, enabling a proactive approach to care.
    - Motivation: Clinicians often have limited time with patients, so NLP-based monitoring outside of sessions can provide critical early warnings. This helps professionals focus on high-risk patients and tailor interventions accordingly, improving the overall quality of mental health care.
-	Students and Academic Institutions
    - Why Focus on This Group: Students face immense academic pressure and social challenges, leading to increased rates of mental health concerns like stress, anxiety, and depression. Educational institutions are increasingly interested in providing mental health support as part of student wellness initiatives.
    - Characteristics: Students, especially in high school and college, frequently use text-based communication through educational platforms, social media, and forums, offering extensive data for NLP tools to analyze.
    - Motivation: Schools and universities could use NLP tools to monitor student well-being, flagging students who may need mental health resources or support. Early identification of mental health risks can help schools proactively address student needs, leading to a healthier academic environment and potentially better academic outcomes.


2.	Requirements Details
We want to build an app to analyze and predict user’s mental health. Users can easily share their stories every day by voice, text, or image and our designed AI system will communicate with them by voice. It will be the same as Snapchat, but we will provide a voice chat feature, so people can feel like they are writing their digital diary. From the context of their talking, writing, or uploading here, we will analyze and classify it into emotion labels to detect their mental health and adjust their appropriate social media content, which will improve their mental health and prevent depression or anxiety issues. In this phase, we will focus on the user interaction and communication, and mental health analysis and classification.

2.1.	User Interaction and Communication
- Story and Diary Sharing: 
    - Users can write daily entries, share personal stories, upload images and log feelings or thoughts in a digital diary.
    - Entries are stored securely with options for tagging (e.g., “happy,” “stressful,” “reflective”) and categorizing.
- Real-Time Conversation with AI: 
    - Users can converse with the AI in real-time, expressing thoughts or emotions through text or speech.
    - The AI can respond appropriately based on user sentiment, creating an empathetic, supportive conversational experience.
- Voice Recognition: 
    - Integrate voice-to-text capabilities to allow users to speak instead of type, providing accessibility for users with typing difficulties or a preference for spoken communication.
    - Speech recognition accuracy should improve with continued use (e.g., via custom vocabularies and personal speech patterns).
- Text-to-Speech (TTS) Response: 
    - The app should have TTS functionality to read out responses, creating a more conversational feel.
    - Allow users to choose different AI voice tones (e.g., calm, energetic) to suit their preferences.

2.2.	Mental Health Analysis and Classification
- Sentiment and Emotion Analysis:
    - Implement NLP algorithms to detect and classify user sentiment (e.g., positive, negative, neutral) and specific emotions (e.g., sadness, joy, anger) in text and speech.
    - Store historical sentiment data to track emotional trends over time, providing insights into the user’s mental health journey.
- Mental Health Classification:
    - Based on language patterns and detected emotions, classify mental health status into relevant categories (e.g., “low stress,” “moderate anxiety,” “high depression risk”).
    - Offer an option for users to view insights, such as how their language correlates with different mental health metrics.
- Conversational Context Recognition:
    - The AI should recognize context to maintain relevant and supportive conversations, adapting responses to the user’s emotional state.
    - Ensure sensitivity to high-risk phrases, such as those that may suggest distress, self-harm, or severe mental health concerns, and provide appropriate responses or resources if needed.

3.	Data Sources and Approach
To create a robust mental health detection system, we’ll gather and analyze data from three main sources—social media text, structured mental health questionnaires, and voice recordings. For social media text, we aim to capture spontaneous expressions of emotions, attitudes, and mental states from social media posts and comments. Using logistic regression, we classify these posts by sentiment (e.g., positive, negative, neutral) and by specific mental health indicators, such as depressive or anxious language patterns, leveraging linguistic features like word choice, polarity, and frequency of specific terms. This helps us detect trends in sentiment and mood over time. In analyzing structured questionnaire data (e.g., PHQ-9, GAD-7), we rely on a decision tree to classify responses based on symptom severity, categorizing users into groups (e.g., mild, moderate, severe) based on their scores. The decision tree’s interpretability is ideal for this structured data, helping the system make straightforward, understandable classifications while considering each response’s influence on overall severity. For voice recordings, we use K-means clustering to identify patterns in vocal tones that correspond with emotional states like calmness, stress, or sadness. The model groups voice features—such as pitch, tone, and speaking rate—into clusters representing different emotional states, which allows for emotion detection even without extensive labeled data. By combining these methods, we create a multi-modal system that cross-validates cues from text, questionnaire responses, and voice patterns, allowing for an adaptive, holistic approach to tracking and analyzing users’ mental health.

4.	Evaluation Metrics
A successful evaluation of the app’s features and their impact on users' mental health would yield several positive outcomes. We would expect high user satisfaction scores averaging 4.5 out of 5 in feedback surveys, with qualitative comments highlighting users' comfort in sharing thoughts and emotions. Utilization rates for the story sharing and real-time conversation features would be at least 75%. Additionally, sentiment analysis would show a measurable improvement, with a 30% increase in positive sentiment in diary entries over three months, alongside a reduction in negative sentiment during interactions with the AI. Performance metrics for the NLP models would exceed 85% accuracy, reflecting their reliability in classifying user emotions. Engagement would be consistent, with users averaging three diary entries per week and a noticeable decrease in high-risk mental health classifications. Users would report greater awareness of their emotions and improved mental health management skills, alongside a sense of community fostered by emotion-aligned content suggestions, collectively demonstrating that the app effectively supports users' mental health through its innovative features.

5.	Key Learning and Impacts
The key learnings from this project encompass several important aspects that can enhance our understanding and approach to mental health technology. Firstly, we will gain insights into the specific mental health challenges faced by our target users, particularly Gen Z, allowing us to tailor our solutions effectively to meet their unique needs. Secondly, the project will deepen our understanding of Natural Language Processing (NLP) and its applications in sentiment analysis, enabling us to develop more accurate and responsive algorithms that can identify emotional cues and mental health indicators from user interactions.

6.	More Information
- [Miro](https://miro.com/app/board/uXjVLIzLbvo=/)
- [Github](https://github.com/MinhHieuBui8386/msba265-finalstorage)
- [Data Management Sheet](https://docs.google.com/spreadsheets/d/1yRUXoIbdwN2IAQY8EbaajiPj86v4ks3mH6Nu0vYP5h4/edit?gid=0#gid=0)

