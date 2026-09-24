import nltk
from nltk.chat.util import Chat, reflections
from flask import Flask, render_template, request

nltk.download('punkt')

pairs = [
    (
        r"(.*) (question)",
        ["With pleasure,say your question",]
    ),
        (
        r"(.*) (Hello)",
        ["Hello,How can i help you.",]
    ),
    (
        r"(.*) (your name| Your name)",
        ["My name is Snow",]
    ),
    (
        r"(.*) (side effects)",
        ["The side effects of a brain tumor can vary widely depending on its size, location, and rate of growth.Some common symptoms may include:Headaches,Nausea and vomiting ,concentration problems,pain,Lack of sleep and other things."]
    ),
    (
        r"(.*) (emergency|severe symptoms)",
        ["In case of an emergency or severe symptoms, seek immediate medical attention by calling emergency services or going to the nearest emergency room.",]
    ),
    (
        r"(.*) (physical activity)",
        ["Engage in walking, aiming for an hour per day. Modify based on your comfort level and consult your healthcare team.",]
    ),
    (
        r"(.*) (Quality of Life:)",
        ["Focus on healthy lifestyle choices, stress management, and maintaining a positive outlook. Regularly discuss your concerns with your healthcare team.",]
    ),
    (
        r"(.*) (recovery)",
        ["Recovery after a brain tumor can vary widely depending on factors such as the type of tumor, its location, the treatment received, and individual health. Here are general steps that might be involved in the recovery process:"
    "1. Medical Follow-up: Regular follow-up appointments with your healthcare team to monitor your progress and address any emerging issues."
    "2. Medication Management: Adherence to prescribed medications for pain management, seizure control, or other specific needs."
    "3. Physical Rehabilitation: Physical therapy to address motor skills, balance, and strength. Occupational therapy can help with daily activities, and speech therapy may be needed for language and communication issues."
    "4. Cognitive Rehabilitation: Exercises and strategies to address cognitive challenges, memory issues, and concentration difficulties."
    "5. Monitoring for Recurrence: Regular medical monitoring to detect any signs of tumor recurrence."
    "6. Lifestyle Changes: Adopting a healthy lifestyle, including a balanced diet, regular exercise, and sufficient sleep, to support overall well-being.",]
    ),
    (
        r"(.*) (epliptic seizure)",
        ["I feel sorry about this. You must now keep the injured person away from anything dangerous next to him and make the person lie on one side to improve his breathing, and you must quickly contact the doctor.",]
    ),
    (
        r"(.*) (fatigue|tiredness|pain|tired)",
        ["You are expected to feel some fatigue during cancer treatment.But if you feel that fatigue persists for weeks and affects your ability to carry out daily tasks, tell your doctor.",]
    ),
    (
        r"(.*) (persistent headaches|headache|unblanced)",
        ["I'm not a doctor, but I can try to provide some general information. I understand that you're concerned about your symptoms. Your symptoms can have various causes, including migraines, stress, or any medication .have you taken any medication?",]
    ),
    (
        r"(.*) (i took|i took medication)",
        [" it's crucial to consider this information when discussing your symptoms with a healthcare professional. Certain medications may have side effects or interactions that could contribute to feelings of headache or unbalance. When you consult with a doctor about your symptoms, make sure to provide a comprehensive list of all the medications you are currently taking",]
    ),
    (
        r"(.*) (i didn’t take)",
        [" Since you're not currently taking any medication, it's still important to discuss your symptoms with a healthcare professional. They can help identify potential causes for your symptoms ",]
    ),
    (
        r"(.*) (stomach ache|stomach|Nausea)",
        ["I'm sorry you're experiencing stomach pain. Have you had any specific foods recently?",]
    ),
    (
        r"(.*) (i ate)",
        ["If you can recall the specific foods you ate before experiencing the stomachache, it might help to note them. Certain foods or ingredients can sometimes trigger digestive issues or discomfort in some individuals. you should mention the types of foods, any spices or sauces used, and whether you've had any changes in your diet recently. This information can be valuable when discussing your symptoms with your healthcare professional",]
    ),
    (
        r"(.*) (i didn’t eat)",
        ["it could be related to various factors beyond food consumption. Other potential causes may include stress, gastrointestinal issues, or even changes in your daily routine. \ ",]
    ),
    (
        r"(.*) (mood|Changes in personality)",
        ["It is normal for you to feel a change in mood during this period of treatment. I suggest that you do any activity like: watch a movie or go out for a walk.",]
    ),
    (
        r"(.*) (thanks|thank you)",
        ["You're welcome! If you have more questions or concerns, feel free to ask.",]
    ),
]

chatbot = Chat(pairs, reflections)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    response = chatbot.respond(user_input)
    return {"response": response}

if __name__ == "__main__":
    app.run(debug=True,port=8000)