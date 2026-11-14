"""All user-visible strings, one dict per language. No machine translation at runtime.

DESCRIPTIONS_ZH covers every sentence artfae.features.aesthetic.describe() can produce:
3 brightness x 3 contrast x 3 colour levels = 27 (enumerated and checked in tests/test_inference.py).
"""
LANGS = {"en": "English", "zh": "繁體中文"}

EMOTION_NAMES = {
    "en": {"amusement": "Amusement", "awe": "Awe", "contentment": "Contentment", "excitement": "Excitement",
           "anger": "Anger", "disgust": "Disgust", "fear": "Fear", "sadness": "Sadness",
           "something_else": "Something else"},
    "zh": {"amusement": "趣味", "awe": "敬畏", "contentment": "滿足", "excitement": "興奮",
           "anger": "憤怒", "disgust": "厭惡", "fear": "恐懼", "sadness": "悲傷", "something_else": "其他"},
}

SENTIMENT_NAMES = {
    "en": {"positive": "Positive", "negative": "Negative", "other": "Other"},
    "zh": {"positive": "正面", "negative": "負面", "other": "其他"},
}

ATTRIBUTE_NAMES = {
    "en": {
        "red": "Mean red", "green": "Mean green", "blue": "Mean blue",
        "hue_sin": "Hue (sine component)", "hue_cos": "Hue (cosine component)",
        "saturation": "Mean saturation", "saturation_var": "Saturation variance", "value": "Mean value (HSV)",
        "warm": "Warm-colour share", "cool": "Cool-colour share", "luminance": "Mean luminance",
        "luminance_std": "Luminance std. dev.", "dark": "Dark-pixel share", "bright": "Bright-pixel share",
        "contrast_p95_p05": "Luminance range (P95 − P5)", "edge_density": "Edge density",
        "edge_tl": "Edges, top left", "edge_tr": "Edges, top right",
        "edge_bl": "Edges, bottom left", "edge_br": "Edges, bottom right",
        **{f"hue_bin_{i}": f"Hue {i * 45}–{i * 45 + 45}°" for i in range(8)},
    },
    "zh": {
        "red": "紅色平均值", "green": "綠色平均值", "blue": "藍色平均值",
        "hue_sin": "色相（正弦分量）", "hue_cos": "色相（餘弦分量）",
        "saturation": "平均飽和度", "saturation_var": "飽和度變異數", "value": "平均明度（HSV）",
        "warm": "暖色比例", "cool": "冷色比例", "luminance": "平均亮度",
        "luminance_std": "亮度標準差", "dark": "暗部比例", "bright": "亮部比例",
        "contrast_p95_p05": "亮度範圍（P95 − P5）", "edge_density": "邊緣密度",
        "edge_tl": "左上邊緣密度", "edge_tr": "右上邊緣密度",
        "edge_bl": "左下邊緣密度", "edge_br": "右下邊緣密度",
        **{f"hue_bin_{i}": f"色相 {i * 45}–{i * 45 + 45}°" for i in range(8)},
    },
}

# every output of describe(), hand-checked
DESCRIPTIONS_ZH = {
    "Low overall brightness, low luminance contrast, and predominantly warm colors.": "整體亮度偏低，明暗對比偏低，色彩以暖色調為主。",
    "Low overall brightness, low luminance contrast, and predominantly cool colors.": "整體亮度偏低，明暗對比偏低，色彩以冷色調為主。",
    "Low overall brightness, low luminance contrast, and mixed or neutral colors.": "整體亮度偏低，明暗對比偏低，色彩冷暖交融或偏中性。",
    "Low overall brightness, moderate luminance contrast, and predominantly warm colors.": "整體亮度偏低，明暗對比適中，色彩以暖色調為主。",
    "Low overall brightness, moderate luminance contrast, and predominantly cool colors.": "整體亮度偏低，明暗對比適中，色彩以冷色調為主。",
    "Low overall brightness, moderate luminance contrast, and mixed or neutral colors.": "整體亮度偏低，明暗對比適中，色彩冷暖交融或偏中性。",
    "Low overall brightness, high luminance contrast, and predominantly warm colors.": "整體亮度偏低，明暗對比強烈，色彩以暖色調為主。",
    "Low overall brightness, high luminance contrast, and predominantly cool colors.": "整體亮度偏低，明暗對比強烈，色彩以冷色調為主。",
    "Low overall brightness, high luminance contrast, and mixed or neutral colors.": "整體亮度偏低，明暗對比強烈，色彩冷暖交融或偏中性。",
    "Moderate overall brightness, low luminance contrast, and predominantly warm colors.": "整體亮度適中，明暗對比偏低，色彩以暖色調為主。",
    "Moderate overall brightness, low luminance contrast, and predominantly cool colors.": "整體亮度適中，明暗對比偏低，色彩以冷色調為主。",
    "Moderate overall brightness, low luminance contrast, and mixed or neutral colors.": "整體亮度適中，明暗對比偏低，色彩冷暖交融或偏中性。",
    "Moderate overall brightness, moderate luminance contrast, and predominantly warm colors.": "整體亮度適中，明暗對比適中，色彩以暖色調為主。",
    "Moderate overall brightness, moderate luminance contrast, and predominantly cool colors.": "整體亮度適中，明暗對比適中，色彩以冷色調為主。",
    "Moderate overall brightness, moderate luminance contrast, and mixed or neutral colors.": "整體亮度適中，明暗對比適中，色彩冷暖交融或偏中性。",
    "Moderate overall brightness, high luminance contrast, and predominantly warm colors.": "整體亮度適中，明暗對比強烈，色彩以暖色調為主。",
    "Moderate overall brightness, high luminance contrast, and predominantly cool colors.": "整體亮度適中，明暗對比強烈，色彩以冷色調為主。",
    "Moderate overall brightness, high luminance contrast, and mixed or neutral colors.": "整體亮度適中，明暗對比強烈，色彩冷暖交融或偏中性。",
    "High overall brightness, low luminance contrast, and predominantly warm colors.": "整體亮度偏高，明暗對比偏低，色彩以暖色調為主。",
    "High overall brightness, low luminance contrast, and predominantly cool colors.": "整體亮度偏高，明暗對比偏低，色彩以冷色調為主。",
    "High overall brightness, low luminance contrast, and mixed or neutral colors.": "整體亮度偏高，明暗對比偏低，色彩冷暖交融或偏中性。",
    "High overall brightness, moderate luminance contrast, and predominantly warm colors.": "整體亮度偏高，明暗對比適中，色彩以暖色調為主。",
    "High overall brightness, moderate luminance contrast, and predominantly cool colors.": "整體亮度偏高，明暗對比適中，色彩以冷色調為主。",
    "High overall brightness, moderate luminance contrast, and mixed or neutral colors.": "整體亮度偏高，明暗對比適中，色彩冷暖交融或偏中性。",
    "High overall brightness, high luminance contrast, and predominantly warm colors.": "整體亮度偏高，明暗對比強烈，色彩以暖色調為主。",
    "High overall brightness, high luminance contrast, and predominantly cool colors.": "整體亮度偏高，明暗對比強烈，色彩以冷色調為主。",
    "High overall brightness, high luminance contrast, and mixed or neutral colors.": "整體亮度偏高，明暗對比強烈，色彩冷暖交融或偏中性。",
}

STRINGS = {
    "en": {
        "app_title": "ArtFAE-Net",
        "app_tagline": "How might viewers feel about this artwork?",
        "language": "Language",
        "page": "Page",
        "page_analyse": "Analyse",
        "page_compare": "Compare",
        "page_about": "About",
        "sidebar_note": "A research demo. Predictions describe how ArtEmis annotators tended to respond, not how you should feel.",
        "device": "Running on: {device}",
        "upload": "Upload an artwork",
        "upload_help": "JPG, PNG or WebP",
        "or_gallery": "Or choose an example from the validation set",
        "gallery_note": "Examples are validation artworks chosen by annotator agreement only; no test images are used.",
        "use_example": "Analyse",
        "spinner": "Looking closely at the artwork…",
        "loading_models": "Loading the model (the first run downloads CLIP, about 1.6 GB)…",
        "inference_time": "Inference: {seconds:.2f} s on {device}",
        "empty_state": "Upload an image or pick an example to begin.",
        "top_emotion": "Most likely emotion",
        "probability": "probability",
        "sentiment": "Sentiment",
        "distribution": "Emotion distribution",
        "prob_axis": "Probability",
        "also_feel": "Emotions viewers may also feel",
        "also_feel_help": "Presence head: the probability that at least one viewer would name this emotion. Shown when above 50%.",
        "also_feel_none": "No other emotion passes 50%.",
        "sentiment_summary": "Sentiment summary",
        "tab_attention": "Attention",
        "tab_aesthetics": "Aesthetics",
        "tab_details": "Details",
        "attention_emotion": "Emotion",
        "attention_opacity": "Overlay opacity",
        "attention_caption": "Model attention, not a validated explanation.",
        "attention_crop": "Shown on the centre square, the region the CLIP encoder sees.",
        "aesthetics_intro": "Deterministic image measurements, not aesthetic judgements. Percentiles compare against the {n:,} training artworks.",
        "group_luminance": "Luminance",
        "group_colour": "Colour & saturation",
        "group_contrast": "Contrast",
        "group_edges": "Edges & composition",
        "group_hue": "Hue histogram",
        "hue_help": "Saturation-weighted share of each 45° hue band.",
        "percentile": "{p:.0f}th percentile",
        "description": "Description fed to the model",
        "annotators": "ArtEmis annotators for this example",
        "annotators_help": "The share of annotators who chose each emotion. The model never saw these labels at inference.",
        "per_seed": "Per-seed top emotions: {items}",
        "artwork_meta": "{painting} · {artist} · {style}",
        "compare_title": "Compare artworks",
        "compare_intro": "Upload 2–4 images, or pick examples, to compare their emotion distributions side by side.",
        "compare_upload": "Upload 2–4 artworks",
        "compare_examples": "Or choose examples",
        "compare_need": "Choose at least two images.",
        "compare_max": "Only the first four images are shown.",
        "compare_chart": "Distributions side by side",
        "about_title": "About this demo",
        "about_method_h": "The method in plain language",
        "about_method": (
            "The model looks at an artwork in three ways at once. A large pre-trained vision model (CLIP ViT-L/14 at "
            "336 px) gives an overall impression and a 7 × 7 grid of regional features. Twenty-eight simple, "
            "deterministic measurements describe colour, brightness, contrast and edges, and a short sentence "
            "summarising them is also read by CLIP's text encoder. ArtFAE-Net fuses these views with a learned gate, "
            "and an attention module lets each emotion look at the image regions that support it.\n\n"
            "The model predicts a distribution over nine emotions, like the spread of answers from a group of "
            "viewers, not a single label. A presence head estimates whether anyone at all would name each "
            "emotion. Because the training set is less balanced than the evaluation data, a fixed prior "
            "correction (no tuning) shifts the logits from the training class frequencies to the validation ones."),
        "about_ensemble": (
            "This demo runs the **pre-registered** model (L/14 R3, λ = 0.3, with prior correction) as a **seed "
            "ensemble**: all five trained seeds (42–46) run on every image and their probabilities are averaged. "
            "The pre-registered test result is the mean over the five individual seeds; the ensemble was reported "
            "as a secondary result (test macro-F1 {ens:.4f})."),
        "about_results_h": "Test results (pre-registered, evaluated once)",
        "about_results_note": "ArtEmis test split, {n:,} artworks; mean ± SD over 5 seeds. Macro-F1 and accuracy use artworks with a unique top emotion; AP measures whether anyone named anger or fear.",
        "about_boot": "Primary vs baseline, paired bootstrap: Δ macro-F1 {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}).",
        "col_model": "Model",
        "col_macro_f1": "Macro-F1",
        "col_acc": "Accuracy",
        "col_ap": "Anger + fear AP",
        "col_neg": "Negative-sentiment F1",
        "row_baseline": "Paper baseline (B/32, paper 14K split)",
        "row_b32": "B/32, R1 (new 43K train)",
        "row_l14_r1": "L/14, R1",
        "row_primary": "L/14, R3 + prior correction (this demo)",
        "about_anger_h": "Known limitation: anger",
        "about_anger": (
            "Anger is the hardest emotion. It is rare as the majority answer (only {n} test artworks) and usually "
            "co-occurs with fear, disgust or sadness. The final model reaches an anger F1 of only {f1:.2f} ± {sd:.2f}. "
            "Treat an 'anger' result as weak evidence, and look at the presence chips and the negative-sentiment "
            "total instead."),
        "about_subjective_h": "Emotions are subjective",
        "about_subjective": (
            "There is no single correct emotion for a painting. The labels come from ArtEmis, where several "
            "annotators each chose the emotion an artwork evoked in them and explained why. The model learns the "
            "distribution of those answers, including the cultural and demographic leanings of that annotator "
            "pool. It cannot know how a particular person will feel."),
        "about_context_h": "Research context",
        "about_context": "ArtFAE-Net is part of a research project in the NTUE TEEP programme (National Taipei University of Education). Author: Nawazish Hussain.",
        "about_privacy": "Images you upload are processed locally on this computer and are not stored.",
        "error_image": "This file could not be read as an image.",
    },
    "zh": {
        "app_title": "ArtFAE-Net",
        "app_tagline": "觀者面對這件作品，可能會有什麼感受？",
        "language": "語言",
        "page": "頁面",
        "page_analyse": "分析",
        "page_compare": "比較",
        "page_about": "關於",
        "sidebar_note": "研究展示。預測結果反映 ArtEmis 標註者的整體反應傾向，並非您應有的感受。",
        "device": "運算裝置：{device}",
        "upload": "上傳藝術作品",
        "upload_help": "支援 JPG、PNG、WebP",
        "or_gallery": "或從驗證集中選擇範例",
        "gallery_note": "範例僅依標註者一致度從驗證集中挑選，未使用任何測試集圖像。",
        "use_example": "分析",
        "spinner": "正在細看這件作品……",
        "loading_models": "正在載入模型（首次執行需下載約 1.6 GB 的 CLIP）……",
        "inference_time": "推論時間：{seconds:.2f} 秒（{device}）",
        "empty_state": "請上傳圖像或選擇範例以開始。",
        "top_emotion": "最可能的情緒",
        "probability": "機率",
        "sentiment": "情感傾向",
        "distribution": "情緒分布",
        "prob_axis": "機率",
        "also_feel": "觀者也可能感受到的情緒",
        "also_feel_help": "存在判斷頭：至少有一位觀者會選擇此情緒的機率。僅顯示高於 50% 者。",
        "also_feel_none": "沒有其他情緒超過 50%。",
        "sentiment_summary": "情感傾向總覽",
        "tab_attention": "注意力",
        "tab_aesthetics": "美學量測",
        "tab_details": "詳細資訊",
        "attention_emotion": "情緒",
        "attention_opacity": "疊圖透明度",
        "attention_caption": "此為模型注意力，並非經驗證的解釋。",
        "attention_crop": "顯示於中央正方形區域，即 CLIP 編碼器實際看到的範圍。",
        "aesthetics_intro": "以下為可重現的圖像量測，而非美感評價。百分位數以 {n:,} 件訓練作品為比較基準。",
        "group_luminance": "亮度",
        "group_colour": "色彩與飽和度",
        "group_contrast": "對比",
        "group_edges": "邊緣與構圖",
        "group_hue": "色相分布",
        "hue_help": "各 45° 色相區間的比例（以飽和度加權）。",
        "percentile": "第 {p:.0f} 百分位",
        "description": "輸入模型的描述文字",
        "annotators": "此範例的 ArtEmis 標註結果",
        "annotators_help": "選擇各情緒的標註者比例。模型推論時不會看到這些標註。",
        "per_seed": "各種子模型的首選情緒：{items}",
        "artwork_meta": "{painting} · {artist} · {style}",
        "compare_title": "比較作品",
        "compare_intro": "上傳 2–4 張圖像或選擇範例，並排比較它們的情緒分布。",
        "compare_upload": "上傳 2–4 件作品",
        "compare_examples": "或選擇範例",
        "compare_need": "請至少選擇兩張圖像。",
        "compare_max": "僅顯示前四張圖像。",
        "compare_chart": "情緒分布並排比較",
        "about_title": "關於本展示",
        "about_method_h": "方法簡介",
        "about_method": (
            "模型同時從三個角度觀看作品。大型預訓練視覺模型（CLIP ViT-L/14，336 像素）提供整體印象，以及 7 × 7 的"
            "區域特徵網格。二十八項簡單且可重現的量測描述色彩、亮度、對比與邊緣，並由這些量測產生一句摘要，交由 CLIP 的"
            "文字編碼器讀取。ArtFAE-Net 以可學習的閘控機制融合這些資訊，注意力模組則讓每種情緒聚焦於支持它的圖像區域。\n\n"
            "模型預測的是九種情緒的機率分布，如同一群觀者答案的分散情形，而非單一標籤。存在判斷頭則估計是否有任何觀者會選擇"
            "某種情緒。由於訓練集的類別比例比評估資料更不平衡，系統以固定的先驗校正（未經調參），將對數機率從訓練集的類別頻率"
            "轉換為驗證集的類別頻率。"),
        "about_ensemble": (
            "本展示執行的是**預先登錄**的模型（L/14 R3，λ = 0.3，含先驗校正），並以**種子集成**方式運行：五個訓練種子"
            "（42–46）的模型都會處理每張圖像，再平均其機率。預先登錄的測試結果是五個單一種子的平均；集成模型則列為次要結果"
            "（測試集 macro-F1 {ens:.4f}）。"),
        "about_results_h": "測試結果（預先登錄，僅評估一次）",
        "about_results_note": "ArtEmis 測試集，共 {n:,} 件作品；數值為 5 個種子的平均 ± 標準差。Macro-F1 與準確率以具唯一首選情緒的作品計算；AP 衡量是否有人選擇憤怒或恐懼。",
        "about_boot": "主要模型對基準模型（成對自助法）：Δ macro-F1 {d:+.3f}（95% 信賴區間 {lo:+.3f} 至 {hi:+.3f}）。",
        "col_model": "模型",
        "col_macro_f1": "Macro-F1",
        "col_acc": "準確率",
        "col_ap": "憤怒＋恐懼 AP",
        "col_neg": "負面情感 F1",
        "row_baseline": "論文基準模型（B/32，論文 14K 分割）",
        "row_b32": "B/32，R1（新 43K 訓練集）",
        "row_l14_r1": "L/14，R1",
        "row_primary": "L/14，R3＋先驗校正（本展示）",
        "about_anger_h": "已知限制：憤怒",
        "about_anger": (
            "憤怒是最難辨識的情緒。它很少成為多數人的答案（測試集中僅 {n} 件作品），且常與恐懼、厭惡或悲傷同時出現。"
            "最終模型的憤怒 F1 僅為 {f1:.2f} ± {sd:.2f}。若結果顯示「憤怒」，請視為微弱證據，並參考存在判斷標籤與負面情感總和。"),
        "about_subjective_h": "情緒是主觀的",
        "about_subjective": (
            "一幅畫沒有唯一正確的情緒。標籤來自 ArtEmis 資料集：多位標註者各自選擇作品帶給他們的情緒並說明原因。模型學到的是"
            "這些答案的分布，也包含該標註者群體的文化與人口背景傾向，因此無法得知某個人實際會有什麼感受。"),
        "about_context_h": "研究背景",
        "about_context": "ArtFAE-Net 為國立臺北教育大學 NTUE TEEP 計畫的研究項目。作者：Nawazish Hussain。",
        "about_privacy": "您上傳的圖像僅在本機處理，不會被儲存。",
        "error_image": "無法將此檔案讀取為圖像。",
    },
}


def t(lang: str, key: str, **kw) -> str:
    s = STRINGS[lang][key]
    return s.format(**kw) if kw else s


def emotion(lang: str, e: str) -> str:
    return EMOTION_NAMES[lang][e]


def sentiment(lang: str, s: str) -> str:
    return SENTIMENT_NAMES[lang][s]


def attribute(lang: str, name: str) -> str:
    return ATTRIBUTE_NAMES[lang][name]


def description(lang: str, text: str) -> str:
    return DESCRIPTIONS_ZH[text] if lang == "zh" else text
