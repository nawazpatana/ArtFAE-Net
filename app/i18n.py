"""All user-visible strings, one dict per language. No machine translation at runtime.

DESCRIPTIONS_ZH covers every sentence artfae.features.aesthetic.describe() can produce:
3 brightness x 3 contrast x 3 colour levels = 27 (enumerated and checked in tests/test_inference.py).
"""
LANGS = {"en": "English", "zh": "繁體中文"}
LANG_SHORT = {"en": "EN", "zh": "繁中"}  # header pill

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
        "model": "Model",
        "model_r1": "Everyday viewers (L/14 R1, recommended)",
        "model_r3": "Rare-emotion lens (L/14 R3)",
        "model_r1_help": "Predicts how a typical group of viewers would split their votes. The most accurate model overall.",
        "model_r3_help": "Looks harder for rare emotions such as anger and fear, but tends to overstate them.",
        "prior_label": "Match validation class frequencies",
        "prior_help": "Optional prior correction: adds log(validation prior) − log(training prior) to the logits before the softmax. Off by default; available for the R1 model only.",
        "model_used": "Model: {name}",
        "model_used_prior": "Model: {name}, with prior correction",
        "also_feel_r3": "From the minority-sensitive model; estimated chance at least one viewer names this emotion.",
        "page": "Page",
        "page_analyse": "Analyse",
        "page_compare": "Compare",
        "page_about": "About",
        "sidebar_note": "A research demo. Predictions describe how ArtEmis annotators tended to respond, not how you should feel.",
        "device": "Running on: {device}",
        "upload": "Upload an artwork",
        "upload_help": "JPG, PNG or WebP",
        "gallery_note": "Examples are validation artworks chosen by annotator agreement only; no test images are used.",
        "spinner": "Looking closely at the artwork…",
        "loading_models": "Loading the model (the first run downloads CLIP, about 1.6 GB)…",
        "inference_time": "Inference: {seconds:.2f} s on {device}",
        "gallery_mode_note": "Live analysis of your own images runs on our research server. This online version shows the example gallery.",
        "precomputed_time": "Precomputed with the full pipeline: {seconds:.2f} s on {device}",
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
            "viewers, not a single label. The minority-sensitive option adds a presence head that estimates whether "
            "at least one viewer would name each emotion. For the default model, an optional prior correction (no "
            "tuning) shifts the logits from the training class frequencies to the validation ones."),
        "about_models": (
            "This demo offers two models. Each is a **seed ensemble**: all five trained seeds (42–46) run on every "
            "image and their probabilities are averaged. Both use checkpoints selected by validation soft "
            "cross-entropy.\n\n"
            "- **Human-vote distribution (L/14 R1), the default:** the paper's recommended model, "
            "`runs/revision_B/step1/R1_s42–46`. The prior correction is optional (off by default).\n"
            "- **Minority-sensitive (L/14 R3):** balanced sampler + presence head (λ = 0.3), "
            "`runs/revision_B/step1/R3_s42–46`. Raw outputs, no prior correction."),
        "about_holdout_h": "Fresh holdout results (pre-registered, evaluated once)",
        "about_holdout_note": "{n:,} ArtEmis artworks never used for training or selection; mean ± SD over 5 seeds, raw outputs. Soft CE measures how well the whole vote distribution is predicted (lower is better).",
        "col_softce": "Soft CE",
        "row_h_r1": "Human-vote distribution (L/14 R1, this demo's default)",
        "row_h_r3": "Minority-sensitive (L/14 R3, this demo)",
        "about_changelog_h": "Changelog",
        "about_changelog": (
            "2026-11-06 · Added the model choice. The default is now L/14 R1 (softCE-selected); the minority-sensitive "
            "option uses the softCE-selected L/14 R3 without prior correction (previously: macro-F1-selected R3 with "
            "prior correction).\n\n"
            "2026-11-06 · Fixed the “Emotions viewers may also feel” values. The presence head was trained with a "
            "positive-class weight (pos_weight) that the app did not undo, which inflated them (on validation, anger "
            "averaged 42% against 9% observed). They are now calibrated: sigmoid(logit − log pos_weight)."),
        "about_results_h": "Earlier test-split results (pre-registered v2, evaluated once)",
        "about_results_note": "ArtEmis test split, {n:,} artworks; mean ± SD over 5 seeds. Macro-F1 and accuracy use artworks with a unique top emotion; AP measures whether anyone named anger or fear.",
        "about_boot": "Previous demo model vs baseline, paired bootstrap: Δ macro-F1 {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}).",
        "col_model": "Model",
        "col_macro_f1": "Macro-F1",
        "col_acc": "Accuracy",
        "col_ap": "Anger + fear AP",
        "col_neg": "Negative-sentiment F1",
        "row_baseline": "Paper baseline (B/32, paper 14K split)",
        "row_b32": "B/32, R1 (new 43K train)",
        "row_l14_r1": "L/14, R1 (macro-F1-selected)",
        "row_primary": "L/14, R3 (macro-F1-selected) + prior correction (previous demo model)",
        "about_anger_h": "Known limitation: anger",
        "about_anger": (
            "Anger is the hardest emotion. It is rare as the majority answer (only {n} test artworks) and usually "
            "co-occurs with fear, disgust or sadness. On the earlier test split the previous demo model reached an "
            "anger F1 of only {f1:.2f} ± {sd:.2f}. Every model gives plain, near-uniform or very dark images more anger "
            "than viewers do, the minority-sensitive model most of all. Treat an 'anger' result as weak evidence, and "
            "look at the negative-sentiment total instead."),
        "about_subjective_h": "Emotions are subjective",
        "about_subjective": (
            "There is no single correct emotion for a painting. The labels come from ArtEmis, where several "
            "annotators each chose the emotion an artwork evoked in them and explained why. The model learns the "
            "distribution of those answers, including the cultural and demographic leanings of that annotator "
            "pool. It cannot know how a particular person will feel."),
        "about_context_h": "Research context",
        "about_context": "ArtFAE-Net is part of a research project in the NTCU TEEP programme (National Taichung University of Education). Author: Nawazish Hussain.",
        "about_privacy": "Images you upload are processed locally on this computer and are not stored.",
        "error_image": "This file could not be read as an image.",
        "error_model": "The model could not be loaded. The online version only shows the example gallery.",
        "model_r1_plain": "Everyday viewers (recommended)",
        "model_r3_plain": "Rare-emotion lens",
        "advanced": "Advanced",
        "prior_plain": "The model learned from training paintings with their own mix of emotions. Turn this on to nudge its answer towards how often each emotion appears among validation paintings. Off by default; only for the everyday-viewers model.",
        "drop_title": "Drop an artwork here",
        "drop_sub": "JPG, PNG or WebP, up to 50 MB",
        "drop_again": "Drop another artwork here",
        "browse": "Browse files",
        "gallery_title": "Or explore the collection",
        "gallery_filter": "What most real viewers felt",
        "filter_all": "All",
        "untitled": "Untitled",
        "card_label": "Analyse {caption}",
        "card_viewers": "Viewers: {emotion}",
        "card_selected": "Showing",
        "ten_people": "If 10 people saw this painting…",
        "ten_people_note": "The predicted split, rounded to whole people.",
        "people_item": "{n} {emotion}",
        "vs_title": "Model vs. real ArtEmis viewers",
        "vs_model": "Model",
        "vs_people": "ArtEmis viewers",
        "vs_note": "A validation artwork: the model never trained on it. The hatched bars show the share of real annotators who chose each emotion.",
        "analyse_another": "Analyse another artwork",
        "alt_upload": "The artwork you uploaded",
        "alt_attention": "Attention heatmap for {emotion} over the centre square of the artwork",
    },
    "zh": {
        "app_title": "ArtFAE-Net",
        "app_tagline": "觀者面對這件作品，可能會有什麼感受？",
        "language": "語言",
        "model": "模型",
        "model_r1": "一般觀者（L/14 R1，建議）",
        "model_r3": "少見情緒視角（L/14 R3）",
        "model_r1_help": "預測一般觀者群體的投票分布，整體最準確的模型。",
        "model_r3_help": "更留意憤怒、恐懼等少見情緒，但往往會高估它們。",
        "prior_label": "校正為驗證集的類別頻率",
        "prior_help": "選用的先驗校正：在 softmax 之前將 log(驗證集先驗) − log(訓練集先驗) 加到對數機率上。預設關閉，僅適用於 R1 模型。",
        "model_used": "模型：{name}",
        "model_used_prior": "模型：{name}（含先驗校正）",
        "also_feel_r3": "來自少數情緒敏感模型；估計至少有一位觀者會選擇此情緒的機率。",
        "page": "頁面",
        "page_analyse": "分析",
        "page_compare": "比較",
        "page_about": "關於",
        "sidebar_note": "研究展示。預測結果反映 ArtEmis 標註者的整體反應傾向，並非您應有的感受。",
        "device": "運算裝置：{device}",
        "upload": "上傳藝術作品",
        "upload_help": "支援 JPG、PNG、WebP",
        "gallery_note": "範例僅依標註者一致度從驗證集中挑選，未使用任何測試集圖像。",
        "spinner": "正在細看這件作品……",
        "loading_models": "正在載入模型（首次執行需下載約 1.6 GB 的 CLIP）……",
        "inference_time": "推論時間：{seconds:.2f} 秒（{device}）",
        "gallery_mode_note": "分析您自己的圖像需在我們的研究伺服器上即時執行。此線上版本展示範例作品集。",
        "precomputed_time": "以完整流程預先計算：{seconds:.2f} 秒（{device}）",
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
            "模型預測的是九種情緒的機率分布，如同一群觀者答案的分散情形，而非單一標籤。少數情緒敏感選項另有存在判斷頭，"
            "估計是否至少有一位觀者會選擇某種情緒。預設模型可選用固定的先驗校正（未經調參），將對數機率從訓練集的類別頻率"
            "轉換為驗證集的類別頻率。"),
        "about_models": (
            "本展示提供兩種模型，皆以**種子集成**方式運行：五個訓練種子（42–46）的模型都會處理每張圖像，再平均其機率。"
            "兩者的檢查點皆依驗證集軟交叉熵挑選。\n\n"
            "- **人類投票分布（L/14 R1，預設）：**論文建議的模型，`runs/revision_B/step1/R1_s42–46`。先驗校正為選用"
            "（預設關閉）。\n"
            "- **少數情緒敏感（L/14 R3）：**平衡取樣＋存在判斷頭（λ = 0.3），`runs/revision_B/step1/R3_s42–46`。"
            "直接輸出，不使用先驗校正。"),
        "about_holdout_h": "新保留集結果（預先登錄，僅評估一次）",
        "about_holdout_note": "{n:,} 件從未用於訓練或挑選的 ArtEmis 作品；數值為 5 個種子的平均 ± 標準差（未校正輸出）。軟交叉熵衡量整體投票分布的預測準確度（越低越好）。",
        "col_softce": "軟交叉熵",
        "row_h_r1": "人類投票分布（L/14 R1，本展示預設）",
        "row_h_r3": "少數情緒敏感（L/14 R3，本展示）",
        "about_changelog_h": "更新紀錄",
        "about_changelog": (
            "2026-11-06 · 新增模型選項。預設改為 L/14 R1（依軟交叉熵挑選）；少數情緒敏感選項使用依軟交叉熵挑選的 "
            "L/14 R3，且不使用先驗校正（先前為依 macro-F1 挑選、含先驗校正的 R3）。\n\n"
            "2026-11-06 · 修正「觀者也可能感受到的情緒」的數值。存在判斷頭以正類權重（pos_weight）訓練，但程式先前未將其"
            "還原，導致數值偏高（在驗證集上，憤怒平均 42%，實際僅 9%）。現已校正為 sigmoid(logit − log pos_weight)。"),
        "about_results_h": "先前測試集結果（第二版預先登錄，僅評估一次）",
        "about_results_note": "ArtEmis 測試集，共 {n:,} 件作品；數值為 5 個種子的平均 ± 標準差。Macro-F1 與準確率以具唯一首選情緒的作品計算；AP 衡量是否有人選擇憤怒或恐懼。",
        "about_boot": "先前展示模型對基準模型（成對自助法）：Δ macro-F1 {d:+.3f}（95% 信賴區間 {lo:+.3f} 至 {hi:+.3f}）。",
        "col_model": "模型",
        "col_macro_f1": "Macro-F1",
        "col_acc": "準確率",
        "col_ap": "憤怒＋恐懼 AP",
        "col_neg": "負面情感 F1",
        "row_baseline": "論文基準模型（B/32，論文 14K 分割）",
        "row_b32": "B/32，R1（新 43K 訓練集）",
        "row_l14_r1": "L/14，R1（依 macro-F1 挑選）",
        "row_primary": "L/14，R3（依 macro-F1 挑選）＋先驗校正（先前展示模型）",
        "about_anger_h": "已知限制：憤怒",
        "about_anger": (
            "憤怒是最難辨識的情緒。它很少成為多數人的答案（測試集中僅 {n} 件作品），且常與恐懼、厭惡或悲傷同時出現。"
            "在先前的測試集上，先前展示模型的憤怒 F1 僅為 {f1:.2f} ± {sd:.2f}。對於單色、近乎均勻或極暗的畫面，所有模型"
            "給出的憤怒機率都高於觀者實際的選擇，少數情緒敏感模型尤甚。若結果顯示「憤怒」，請視為微弱證據，並參考負面情感總和。"),
        "about_subjective_h": "情緒是主觀的",
        "about_subjective": (
            "一幅畫沒有唯一正確的情緒。標籤來自 ArtEmis 資料集：多位標註者各自選擇作品帶給他們的情緒並說明原因。模型學到的是"
            "這些答案的分布，也包含該標註者群體的文化與人口背景傾向，因此無法得知某個人實際會有什麼感受。"),
        "about_context_h": "研究背景",
        "about_context": "ArtFAE-Net 為國立臺中教育大學 NTCU TEEP 計畫的研究項目。作者：Nawazish Hussain。",
        "about_privacy": "您上傳的圖像僅在本機處理，不會被儲存。",
        "error_image": "無法將此檔案讀取為圖像。",
        "error_model": "無法載入模型。線上版本僅展示範例作品集。",
        "model_r1_plain": "一般觀者（建議）",
        "model_r3_plain": "少見情緒視角",
        "advanced": "進階設定",
        "prior_plain": "模型是從訓練集作品學習的，而訓練集有它自己的情緒比例。開啟後，答案會稍微調整為各情緒在驗證集作品中出現的頻率。預設關閉，僅適用於一般觀者模型。",
        "drop_title": "將藝術作品拖放到這裡",
        "drop_sub": "支援 JPG、PNG、WebP，最大 50 MB",
        "drop_again": "拖放另一件作品到這裡",
        "browse": "選擇檔案",
        "gallery_title": "或探索館藏範例",
        "gallery_filter": "多數真實觀者的感受",
        "filter_all": "全部",
        "untitled": "無題",
        "card_label": "分析 {caption}",
        "card_viewers": "觀者：{emotion}",
        "card_selected": "目前顯示",
        "ten_people": "如果有 10 個人看到這幅畫……",
        "ten_people_note": "依預測分布換算成整數人數。",
        "people_item": "{emotion} {n} 人",
        "vs_title": "模型與真實 ArtEmis 觀者對照",
        "vs_model": "模型",
        "vs_people": "ArtEmis 觀者",
        "vs_note": "這是驗證集作品，模型從未以它訓練。斜線長條為真實標註者選擇各情緒的比例。",
        "analyse_another": "分析其他作品",
        "alt_upload": "您上傳的作品",
        "alt_attention": "「{emotion}」的注意力熱圖，疊於作品的中央正方形區域",
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
