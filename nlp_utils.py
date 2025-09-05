# Enhanced NLP utils for leaf disease treatment advice
def generate_treatment_summary(disease, lang="English"):
    treatments = {
        "early_blight": {
            "English": "Early blight requires immediate attention. Remove all infected leaves and dispose of them properly. Apply copper-based fungicide or neem oil spray every 7-14 days. Improve air circulation around plants and avoid overhead watering. Consider using resistant varieties in future plantings.",
            "Hindi": "अर्ली ब्लाइट पर तुरंत ध्यान देने की जरूरत है। सभी संक्रमित पत्तियों को हटाकर उचित तरीके से नष्ट करें। हर 7-14 दिन में कॉपर आधारित फफूंदीनाशक या नीम तेल का छिड़काव करें। पौधों के चारों ओर हवा का संचार बढ़ाएं।",
            "Spanish": "El tizón temprano requiere atención inmediata. Retire todas las hojas infectadas y deséchelas adecuadamente. Aplique fungicida a base de cobre o aceite de neem cada 7-14 días. Mejore la circulación de aire alrededor de las plantas.",
            "French": "La brûlure précoce nécessite une attention immédiate. Retirez toutes les feuilles infectées et éliminez-les correctement. Appliquez un fongicide à base de cuivre ou de l'huile de neem tous les 7-14 jours."
        },
        "late_blight": {
            "English": "Late blight is a serious fungal disease. Immediately remove and destroy infected plant parts. Apply preventive copper-based sprays or systemic fungicides. Ensure good drainage and avoid watering leaves directly. In severe cases, consider removing entire plants to prevent spread.",
            "Hindi": "लेट ब्लाइट एक गंभीर फफूंदी रोग है। तुरंत संक्रमित भागों को हटाकर नष्ट करें। कॉपर आधारित स्प्रे या सिस्टमिक फफूंदीनाशक का प्रयोग करें। अच्छी जल निकासी सुनिश्चित करें।",
            "Spanish": "El tizón tardío es una enfermedad fúngica grave. Retire inmediatamente y destruya las partes infectadas. Aplique aerosoles preventivos a base de cobre o fungicidas sistémicos. Asegure un buen drenaje.",
            "French": "Le mildiou tardif est une maladie fongique grave. Retirez immédiatement et détruisez les parties infectées. Appliquez des pulvérisations préventives à base de cuivre ou des fongicides systémiques."
        },
        "healthy": {
            "English": "Great news! Your plant appears healthy. Continue with regular care: water at soil level, ensure good air circulation, monitor regularly for early signs of disease, and maintain proper nutrition with balanced fertilizer.",
            "Hindi": "बहुत अच्छी खबर! आपका पौधा स्वस्थ दिख रहा है। नियमित देखभाल जारी रखें: मिट्टी के स्तर पर पानी दें, अच्छी हवा सुनिश्चित करें, और संतुलित खाद से उचित पोषण बनाए रखें।",
            "Spanish": "¡Excelentes noticias! Su planta parece saludable. Continúe con el cuidado regular: riegue a nivel del suelo, asegure una buena circulación de aire y mantenga una nutrición adecuada.",
            "French": "Excellentes nouvelles ! Votre plante semble en bonne santé. Continuez avec des soins réguliers : arrosez au niveau du sol, assurez une bonne circulation d'air et maintenez une nutrition appropriée."
        }
    }

    return treatments.get(disease, {}).get(lang, "No specific advice available for this condition.")

def get_prevention_tips(lang="English"):
    tips = {
        "English": [
            "Water plants at soil level, not on leaves",
            "Ensure proper spacing for air circulation",
            "Remove plant debris regularly",
            "Use disease-resistant varieties when possible",
            "Apply preventive fungicide treatments during humid weather"
        ],
        "Hindi": [
            "पत्तियों पर नहीं, मिट्टी के स्तर पर पानी दें",
            "हवा के संचार के लिए उचित दूरी बनाए रखें",
            "नियमित रूप से पौधों का मलबा हटाएं",
            "जब संभव हो तो रोग प्रतिरोधी किस्मों का उपयोग करें"
        ]
    }

    return tips.get(lang, tips["English"])
