"""Study strings (English / 繁體中文). Emotion and sentiment names come from the demo's i18n (app/i18n.py).

The zh-TW SUS items are this project's translation; replace them with a published, validated Traditional Chinese
SUS before collecting real data, and keep the item order.
"""
STRINGS = {
    "en": {
        "study_title": "How does this artwork make you feel?",
        "study_tagline": "A research study on emotions and artworks",
        "progress": "Artwork {i} of {n}",
        "practice_badge": "Practice",
        "language": "Language",
        "pilot_banner": "Pilot test: answers are stored, but used in research only if you opt in and an ethics committee approves it.",
        "config_error": "This study is not available: its deployment settings are incomplete or unsafe. Please contact the researcher.",
        "frozen_error": "This study is not available: its files do not match the frozen stimulus set. Please contact the researcher.",
        "alt_artwork": "The artwork for this question",
        "alt_attention": "The same artwork with the model's attention for “{emotion}” shown as a bright overlay",
        # consent
        "consent_h": "Before you begin",
        "consent_check": "I am 18 or older, I have read this information, and I agree to take part.",
        "consent_check_pilot": "I am 18 or older, I have read this information, and I agree to take part in this pilot.",
        "consent_optin": "If a research ethics committee later approves it, my anonymous pilot answers may be used in research and shared in an anonymised research dataset.",
        "consent_continue": "Continue",
        "consent_required": "Please tick the box to agree before continuing.",
        # participant code (shown once, right after consent)
        "pcode_h": "Thank you for agreeing to take part",
        "pcode_label": "Your participant code:",
        "pcode_body": "Please write it down; you need it to withdraw your answers.",
        "pcode_continue": "I have written it down. Continue",
        "pcode_reminder": "Your participant code (needed to withdraw your answers): {code}",
        # instructions
        "instr_h": "How it works",
        "instr_body": (
            "You will see 22 artworks, one at a time. For most of them:\n\n"
            "1. **Your feeling.** Look at the artwork and choose the emotion you feel most, and how strongly. "
            "There are no right or wrong answers.\n"
            "2. **A computer's guess.** We then show how a computer model expects a group of people to feel, and "
            "ask how well that matches.\n"
            "3. **Two versions.** You compare two versions of the guess and pick the one that seems more true.\n"
            "4. **Optional: where it looked.** You may rate a highlighted area of the image.\n\n"
            "Two of the 22 screens only ask for your feeling and contain an instruction, to check that answers are "
            "read carefully. It takes about 20–25 minutes. First, a short practice round."),
        "instr_start_practice": "Start the practice round",
        "practice_done_h": "Practice complete",
        "practice_done_body": "Well done. The study itself starts now; your answers from here on are recorded.",
        "start_study": "Start the study",
        # blind
        "blind_q": "Which emotion do you feel most?",
        "blind_strength": "How strongly do you feel it?",
        "strength_1": "1 · Very weakly",
        "strength_5": "5 · Very strongly",
        "blind_reason": "Why? (optional)",
        "blind_reason_ph": "A few words about what in the artwork makes you feel this way",
        "submit": "Submit",
        "skip_artwork": "Skip this artwork",
        "next": "Next",
        "need_emotion": "Please choose an emotion.",
        "need_strength": "Please choose how strongly you feel it.",
        "check_instruction": "Attention check: for this artwork, please choose “{emotion}” and strength {strength}.",
        # reveal
        "reveal_h": "How a group of people might feel",
        "reveal_intro": "A computer model's estimate for this artwork, after studying many people's reactions to other artworks.",
        "ten_people": "If 10 people saw this artwork…",
        "distribution": "All nine emotions",
        "plaus_q": "How well does this match how a group of people would feel?",
        "plaus_1": "1 · Not at all",
        "plaus_5": "5 · Very well",
        "need_rating": "Please choose a rating.",
        # A/B
        "ab_h": "Two versions",
        "ab_intro": "Here are two different estimates for the same artwork.",
        "ab_a": "Version A",
        "ab_b": "Version B",
        "ab_q": "Which better describes how people would feel?",
        "ab_opt_a": "Version A",
        "ab_opt_b": "Version B",
        "ab_opt_same": "No difference",
        "need_choice": "Please choose A, B or no difference.",
        # attention
        "att_h": "Where the model looked (optional)",
        "att_intro": "The bright area is where the model looked most when it estimated “{emotion}”. It is the model's focus, not an explanation.",
        "att_q": "Does the highlighted area relate to what you feel?",
        "att_1": "1 · Not at all",
        "att_5": "5 · Very much",
        "att_skip": "Skip this question",
        # questionnaire
        "q_h": "About you",
        "q_intro": "A few short questions. Questions marked optional can be left blank.",
        "q_age": "Age",
        "q_gender": "Gender (optional)",
        "q_art": "Art experience",
        "q_country": "Country or culture you most identify with",
        "q_native": "Native language",
        "q_native_other": "If other, which language? (optional)",
        "q_sus_h": "About this study's interface",
        "q_sus_intro": "Rate each statement from 1 (strongly disagree) to 5 (strongly agree). “The system” means the screens that showed the computer's estimates.",
        "q_comment": "Any other comments? (optional)",
        "q_submit": "Finish",
        "q_missing": "Please answer: {items}.",
        "age_18_24": "18–24", "age_25_34": "25–34", "age_35_44": "35–44", "age_45_54": "45–54",
        "age_55_64": "55–64", "age_65": "65 or over", "age_na": "Prefer not to say",
        "gender_woman": "Woman", "gender_man": "Man", "gender_other": "Another gender", "gender_na": "Prefer not to say",
        "art_none": "None", "art_some": "Some (classes, regular museum visits)", "art_trained": "Trained (art degree or professional)",
        "lang_zh": "Mandarin Chinese", "lang_nan": "Taiwanese (Hokkien)", "lang_hak": "Hakka", "lang_en": "English",
        "lang_indig": "A Formosan (Indigenous) language", "lang_other": "Other",
        "sus_scale_1": "1 · Strongly disagree", "sus_scale_5": "5 · Strongly agree",
        # thanks
        "thanks_h": "Thank you!",
        "thanks_body": "Your answers have been saved. If you were asked for a completion code, here it is:",
        "thanks_close": "You can now close this page.",
        "error_storage": "Your answer could not be saved. Please try again; if it keeps happening, contact the researcher.",
    },
    "zh": {
        "study_title": "這件作品帶給您什麼感受？",
        "study_tagline": "一項關於情緒與藝術作品的研究",
        "progress": "第 {i} 件，共 {n} 件",
        "practice_badge": "練習",
        "language": "語言",
        "pilot_banner": "試行測試：回答會被儲存，但只有在您同意且經研究倫理審查委員會核准後，才會用於研究。",
        "config_error": "本研究目前無法使用：部署設定不完整或不安全。請聯絡研究人員。",
        "frozen_error": "本研究目前無法使用：檔案與凍結的刺激材料不一致。請聯絡研究人員。",
        "alt_artwork": "本題的藝術作品",
        "alt_attention": "同一件作品，以亮色疊圖標示模型推測「{emotion}」時的注意力",
        "consent_h": "開始之前",
        "consent_check": "我已年滿 18 歲，已閱讀以上說明，並同意參與本研究。",
        "consent_check_pilot": "我已年滿 18 歲，已閱讀以上說明，並同意參與本次試行測試。",
        "consent_optin": "若日後經研究倫理審查委員會核准，我同意我在試行中的匿名回答可用於研究，並納入去識別化的研究資料集中分享。",
        "consent_continue": "繼續",
        "consent_required": "請先勾選同意後再繼續。",
        "pcode_h": "感謝您同意參與",
        "pcode_label": "您的參與者代碼：",
        "pcode_body": "請把它記下來；撤回您的回答時需要這組代碼。",
        "pcode_continue": "我已記下代碼，繼續",
        "pcode_reminder": "您的參與者代碼（撤回回答時需要）：{code}",
        "instr_h": "研究流程",
        "instr_body": (
            "您將逐一看到 22 件藝術作品。大部分作品的流程如下：\n\n"
            "1. **您的感受。**觀看作品，選出您最強烈的情緒以及強度。答案沒有對錯。\n"
            "2. **電腦的推測。**接著我們會顯示電腦模型推測一群人可能的感受，並請您評估是否符合。\n"
            "3. **兩個版本。**比較兩個版本的推測，選出您覺得較接近真實的一個。\n"
            "4. **選答：模型關注的區域。**您可以為圖中標示的區域評分。\n\n"
            "22 個畫面中有兩個只詢問您的感受，並附有指示，用來確認您有仔細閱讀。全程約 20–25 分鐘。首先是一回練習。"),
        "instr_start_practice": "開始練習",
        "practice_done_h": "練習完成",
        "practice_done_body": "做得很好。正式研究現在開始，接下來的回答都會被記錄。",
        "start_study": "開始正式研究",
        "blind_q": "您最強烈感受到的是哪一種情緒？",
        "blind_strength": "這種感受有多強烈？",
        "strength_1": "1 · 非常微弱",
        "strength_5": "5 · 非常強烈",
        "blind_reason": "原因（選填）",
        "blind_reason_ph": "簡單寫下作品中讓您有這種感受的地方",
        "submit": "送出",
        "skip_artwork": "略過這件作品",
        "next": "下一步",
        "need_emotion": "請選擇一種情緒。",
        "need_strength": "請選擇感受的強度。",
        "check_instruction": "注意力檢查：這件作品請選擇「{emotion}」，強度選 {strength}。",
        "reveal_h": "一群人可能的感受",
        "reveal_intro": "電腦模型在研究許多人對其他作品的反應後，對這件作品所做的推測。",
        "ten_people": "如果有 10 個人看到這件作品……",
        "distribution": "九種情緒",
        "plaus_q": "這與一群人實際的感受相符程度如何？",
        "plaus_1": "1 · 完全不符",
        "plaus_5": "5 · 非常符合",
        "need_rating": "請選擇一個評分。",
        "ab_h": "兩個版本",
        "ab_intro": "以下是對同一件作品的兩種不同推測。",
        "ab_a": "版本 A",
        "ab_b": "版本 B",
        "ab_q": "哪一個比較能描述人們的感受？",
        "ab_opt_a": "版本 A",
        "ab_opt_b": "版本 B",
        "ab_opt_same": "沒有差別",
        "need_choice": "請選擇 A、B 或沒有差別。",
        "att_h": "模型關注的區域（選答）",
        "att_intro": "亮處是模型推測「{emotion}」時最關注的區域。這只是模型的焦點，並非解釋。",
        "att_q": "標示的區域與您的感受有關嗎？",
        "att_1": "1 · 完全無關",
        "att_5": "5 · 非常相關",
        "att_skip": "略過此題",
        "q_h": "關於您",
        "q_intro": "幾個簡短的問題。標示「選填」的題目可以不填。",
        "q_age": "年齡",
        "q_gender": "性別（選填）",
        "q_art": "藝術經驗",
        "q_country": "您最認同的國家或文化",
        "q_native": "母語",
        "q_native_other": "若為其他，請填寫語言（選填）",
        "q_sus_h": "關於本研究的介面",
        "q_sus_intro": "請為每一句敘述評分，1 表示非常不同意，5 表示非常同意。「這個系統」指的是顯示電腦推測的畫面。",
        "q_comment": "其他意見（選填）",
        "q_submit": "完成",
        "q_missing": "請回答：{items}。",
        "age_18_24": "18–24 歲", "age_25_34": "25–34 歲", "age_35_44": "35–44 歲", "age_45_54": "45–54 歲",
        "age_55_64": "55–64 歲", "age_65": "65 歲以上", "age_na": "不願透露",
        "gender_woman": "女性", "gender_man": "男性", "gender_other": "其他性別", "gender_na": "不願透露",
        "art_none": "無", "art_some": "有一些（上過課、常去美術館）", "art_trained": "受過訓練（藝術學位或專業）",
        "lang_zh": "華語", "lang_nan": "臺灣台語", "lang_hak": "客語", "lang_en": "英語",
        "lang_indig": "臺灣原住民族語", "lang_other": "其他",
        "sus_scale_1": "1 · 非常不同意", "sus_scale_5": "5 · 非常同意",
        "thanks_h": "謝謝您！",
        "thanks_body": "您的回答已儲存。若需要完成代碼，請使用：",
        "thanks_close": "您現在可以關閉此頁面。",
        "error_storage": "無法儲存您的回答，請再試一次；若持續發生，請聯絡研究人員。",
    },
}

# System Usability Scale (Brooke, 1996), standard item order; odd items positive, even items negative
SUS = {
    "en": [
        "I think that I would like to use this system frequently.",
        "I found the system unnecessarily complex.",
        "I thought the system was easy to use.",
        "I think that I would need the support of a technical person to be able to use this system.",
        "I found the various functions in this system were well integrated.",
        "I thought there was too much inconsistency in this system.",
        "I would imagine that most people would learn to use this system very quickly.",
        "I found the system very cumbersome to use.",
        "I felt very confident using the system.",
        "I needed to learn a lot of things before I could get going with this system.",
    ],
    "zh": [  # project translation; replace with a validated Traditional Chinese SUS before real data collection
        "我想我會願意經常使用這個系統。",
        "我覺得這個系統過於複雜。",
        "我覺得這個系統很容易使用。",
        "我想我需要技術人員的協助才能使用這個系統。",
        "我覺得這個系統的各項功能整合得很好。",
        "我覺得這個系統有太多不一致的地方。",
        "我想大部分的人都能很快學會使用這個系統。",
        "我覺得這個系統用起來很笨拙。",
        "使用這個系統時，我覺得很有信心。",
        "在開始使用這個系統之前，我需要學習很多東西。",
    ],
}

AGE = ["age_18_24", "age_25_34", "age_35_44", "age_45_54", "age_55_64", "age_65", "age_na"]
GENDER = ["gender_woman", "gender_man", "gender_other", "gender_na"]
ART = ["art_none", "art_some", "art_trained"]
NATIVE = ["lang_zh", "lang_nan", "lang_hak", "lang_en", "lang_indig", "lang_other"]


def t(lang: str, key: str, **kw) -> str:
    s = STRINGS[lang][key]
    return s.format(**kw) if kw else s
