import streamlit as st

from modules.ui_components import apply_global_style, render_feature_grid, render_page_hero, render_section_header


st.set_page_config(page_title="心理学理论框架 - PetEcho", page_icon="🐾", layout="wide")

apply_global_style("theory")

st.markdown(
    """
<style>
.block-container {
    max-width: 1240px;
}

.page-hero {
    margin-top: 1rem;
}

.section-head {
    margin-top: 2.05rem;
    margin-bottom: 1.05rem;
}

.feature-grid {
    gap: 18px;
    margin: 1.05rem 0 2.15rem;
}

.feature-card,
.explain-card {
    border-color: rgba(226, 156, 126, 0.36);
    background: rgba(255, 255, 255, 0.74);
    box-shadow: 0 12px 28px rgba(86, 61, 48, 0.07);
}

.theory-intro-panel {
    margin: 1.1rem 0 2.25rem;
    padding: 18px 20px;
    border-radius: 16px;
    border: 1px solid rgba(226, 156, 126, 0.38);
    border-left: 5px solid #ef8796;
    background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,245,239,0.7) 55%, rgba(241,250,241,0.68));
    color: #6b5048;
    line-height: 1.82;
    box-shadow: 0 12px 28px rgba(86, 61, 48, 0.07);
}

.theory-split {
    display: grid;
    grid-template-columns: minmax(0, 1.08fr) minmax(0, 0.92fr);
    gap: 26px;
    margin: 2.2rem 0 2.35rem;
}

.theory-panel {
    border-radius: 18px;
    padding: 22px 24px;
    border: 1px solid rgba(226, 156, 126, 0.42);
    background:
        rgba(255, 255, 255, 0.72);
    box-shadow: 0 14px 30px rgba(86, 61, 48, 0.08);
}

.theory-panel.green {
    border-color: rgba(169, 201, 175, 0.62);
    background:
        rgba(255, 255, 255, 0.72);
}

.theory-panel-title {
    color: #65372f;
    font-size: 1.22rem;
    font-weight: 950;
    margin-bottom: 4px;
}

.theory-panel-desc {
    color: #80655e;
    line-height: 1.72;
    margin-bottom: 16px;
}

.theory-flow-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.theory-flow-step {
    min-height: 92px;
    border-radius: 14px;
    border: 1px solid rgba(226, 156, 126, 0.34);
    background: rgba(255, 250, 246, 0.72);
    padding: 14px 15px;
}

.step-kicker {
    color: #df735b;
    font-weight: 950;
    font-size: 0.84rem;
    margin-bottom: 5px;
}

.step-title {
    color: #603b34;
    font-weight: 950;
    margin-bottom: 3px;
}

.step-body {
    color: #80655e;
    font-size: 0.92rem;
    line-height: 1.62;
}

.boundary-card {
    display: grid;
    grid-template-columns: 38px minmax(0, 1fr);
    gap: 12px;
    align-items: start;
    border-radius: 14px;
    border: 1px solid rgba(226, 156, 126, 0.34);
    background: rgba(255, 250, 246, 0.74);
    padding: 14px 15px;
    margin-top: 12px;
}

.boundary-mark {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    color: #75463b;
    font-weight: 950;
    background: linear-gradient(135deg, #ffe1c8, #eef7f1);
}

.strategy-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin: 1rem 0 2.2rem;
}

.strategy-card {
    min-height: 175px;
    border-radius: 16px;
    border: 1px solid rgba(226, 156, 126, 0.34);
    background: rgba(255, 255, 255, 0.74);
    padding: 18px;
    box-shadow: 0 12px 28px rgba(86, 61, 48, 0.07);
}

.strategy-tag {
    display: inline-flex;
    padding: 6px 10px;
    border-radius: 999px;
    color: #79483d;
    background: rgba(255, 247, 239, 0.86);
    border: 1px solid rgba(226, 156, 126, 0.36);
    font-size: 0.84rem;
    font-weight: 900;
    margin-bottom: 12px;
}

.strategy-title {
    color: #603b34;
    font-weight: 950;
    margin-bottom: 8px;
}

.strategy-body {
    color: #80655e;
    line-height: 1.72;
    font-size: 0.94rem;
}

@media (max-width: 980px) {
    .theory-split,
    .strategy-grid {
        grid-template-columns: 1fr;
    }
    .theory-flow-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

render_page_hero(
    "心理学理论框架",
    "本页面概述 PetEcho 的心理学支持框架：系统在陪伴、纪念、现实支持和安全边界之间形成可解释的工作流程。项目不进行临床诊断，也不制造宠物复活的暗示，而是帮助用户更温和地整理哀伤与记忆。",
    eyebrow="PetEcho 支持框架",
    badges=["哀伤支持", "健康联结", "意义整理", "安全优先"],
)

render_section_header("项目心理学定位", "PetEcho 以宠物离别后的情绪表达、记忆整理和低负担支持为核心。", "✦")
st.markdown(
    """
<div class="theory-intro-panel">
系统会先分析输入中的情绪负荷、哀伤阶段倾向和安全风险，再决定回应重点。AI 生成内容受到宠物档案、记忆检索结果、心理支持策略和伦理边界约束，以减少空泛安慰、事实偏离和过度沉浸式互动。
</div>
""",
    unsafe_allow_html=True,
)

render_section_header("理论基础", "以下理论为系统的回应策略、纪念功能和安全边界提供依据。", "♡")
render_feature_grid(
    [
        {
            "icon": "1",
            "title": "哀伤双进程模型",
            "body": "该模型认为哀伤个体会在“面对失去”和“恢复生活”之间摆动。PetEcho 因此既支持记忆表达，也在合适时提供回到当下的小行动。",
        },
        {
            "icon": "2",
            "title": "持续性联结理论",
            "body": "该理论强调健康哀伤并不一定要切断情感联结。PetEcho 通过照片、记忆卡和纪念空间保留象征性联结，同时避免复活式表达。",
        },
        {
            "icon": "3",
            "title": "意义重建理论",
            "body": "该理论关注丧失后个体如何重新理解关系和生活意义。PetEcho 将散乱回忆整理为可保存的纪念内容，帮助关系意义被重新安放。",
        },
        {
            "icon": "4",
            "title": "行为激活",
            "body": "行为激活强调通过具体、可完成的小行动减少退缩。PetEcho 在低落或功能受损时推荐低负担练习，如补充水分、联系他人或写下短句。",
        },
    ]
)

st.markdown(
    """
<div class="theory-split">
    <section class="theory-panel">
        <div class="theory-panel-title">系统如何形成回应</div>
        <div class="theory-panel-desc">系统回复由多个模块共同生成，包括状态理解、风险判断、策略选择和记忆调用控制。</div>
        <div class="theory-flow-grid">
            <div class="theory-flow-step">
                <div class="step-kicker">01</div>
                <div class="step-title">表达感受</div>
                <div class="step-body">用户输入可包含思念、愧疚、愤怒、麻木、求助或其他情绪表达。</div>
            </div>
            <div class="theory-flow-step">
                <div class="step-kicker">02</div>
                <div class="step-title">理解状态</div>
                <div class="step-body">系统识别情绪负荷、哀伤阶段倾向，以及是否存在安全风险。</div>
            </div>
            <div class="theory-flow-step">
                <div class="step-kicker">03</div>
                <div class="step-title">选择支持方式</div>
                <div class="step-body">根据当下需要，在承接情绪、现实支持、意义整理之间调整重点。</div>
            </div>
            <div class="theory-flow-step">
                <div class="step-kicker">04</div>
                <div class="step-title">谨慎调用记忆</div>
                <div class="step-body">只有当当前表达与回忆线索贴合时，系统才会轻量调用宠物记忆。</div>
            </div>
            <div class="theory-flow-step">
                <div class="step-kicker">05</div>
                <div class="step-title">生成回应</div>
                <div class="step-body">回复会尽量温暖、具体、不过度展开，也不编造没有证据的细节。</div>
            </div>
            <div class="theory-flow-step">
                <div class="step-kicker">06</div>
                <div class="step-title">回到现实</div>
                <div class="step-body">在合适场景下，系统会提供低负担行动建议，帮助支持延伸到现实生活。</div>
            </div>
        </div>
    </section>
    <section class="theory-panel green">
        <div class="theory-panel-title">使用边界</div>
        <div class="theory-panel-desc">PetEcho 的目标是提供温和支持，而不是替代现实关系或专业帮助。</div>
        <div class="boundary-card">
            <div class="boundary-mark">1</div>
            <div><div class="step-title">不替代心理咨询</div><div class="step-body">系统支持情绪表达和记忆整理，但不进行临床诊断。</div></div>
        </div>
        <div class="boundary-card">
            <div class="boundary-mark">2</div>
            <div><div class="step-title">不制造复活错觉</div><div class="step-body">纪念形象和对话只用于安放想念，不代表宠物真实回到身边。</div></div>
        </div>
        <div class="boundary-card">
            <div class="boundary-mark">3</div>
            <div><div class="step-title">高风险先保护安全</div><div class="step-body">当表达中出现自伤、自杀或即时危险信号，会优先引导联系现实支持。</div></div>
        </div>
        <div class="boundary-card">
            <div class="boundary-mark">4</div>
            <div><div class="step-title">不强行唤起回忆</div><div class="step-body">当表达更需要稳定支持时，系统会减少记忆展开，优先降低情绪负荷。</div></div>
        </div>
    </section>
</div>
""",
    unsafe_allow_html=True,
)

render_section_header("不同状态下的支持重点", "系统会根据表达内容调整回应重心，避免所有场景都使用同一种安慰方式。", "✓")
st.markdown(
    """
<div class="strategy-grid">
    <div class="strategy-card">
        <div class="strategy-tag">强烈思念</div>
        <div class="strategy-title">先承接想念</div>
        <div class="strategy-body">系统先承接思念和失落，只在记忆线索贴合时引入一小段温暖回忆，避免过度展开沉重细节。</div>
    </div>
    <div class="strategy-card">
        <div class="strategy-tag">愧疚或后悔</div>
        <div class="strategy-title">松动过度自责</div>
        <div class="strategy-body">系统先承认遗憾感的存在，再引导看见曾经的照顾与付出，减少单纯劝说式安慰。</div>
    </div>
    <div class="strategy-card">
        <div class="strategy-tag">麻木或低落</div>
        <div class="strategy-title">回到可做的一小步</div>
        <div class="strategy-body">系统减少复杂分析，提供低负担现实动作，帮助情绪支持从对话延伸到日常照顾。</div>
    </div>
    <div class="strategy-card">
        <div class="strategy-tag">安全风险</div>
        <div class="strategy-title">优先现实支持</div>
        <div class="strategy-body">当出现自伤、自杀或即时危险信号时，系统暂停普通纪念式互动，优先引导现实求助。</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
