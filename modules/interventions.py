def recommend_intervention(meta: dict | None) -> dict:
    meta = meta or {}
    emotion = meta.get("emotion", {}) or {}
    grief_stage = meta.get("grief_stage", "yearning")
    risk = meta.get("risk", {}) or {}
    strategy = meta.get("strategy", {}) or {}
    risk_level = risk.get("level", "low")

    if risk_level in {"high", "imminent"}:
        return {
            "key": "safety_contact",
            "title": "先让现实中的人靠近你",
            "category": "安全支持",
            "prompt": "请写下一个现在可以联系的人，或一个你可以立刻拨打的求助电话。",
            "placeholder": "例如：给朋友发一句“我现在不太安全，能不能陪我一下？”",
            "rationale": "当安全风险较高时，系统会优先帮助用户连接现实支持，而不是继续普通回忆练习。",
            "allow_save": True,
        }

    if risk_level == "medium":
        return {
            "key": "reality_anchor",
            "title": "先找到一个现实里的支点",
            "category": "稳定支持",
            "prompt": "写下此刻最容易完成的一件照顾自己的小事，或一个可以联系的人。",
            "placeholder": "例如：先喝一点水，然后给朋友发一句“我今天状态很差，能不能陪我聊十分钟？”",
            "rationale": "当痛苦已经影响睡眠、进食或日常功能时，练习会优先帮助用户连接现实支持和基本照顾。",
            "allow_save": True,
        }

    sadness = emotion.get("sadness", 0.0)
    loneliness = emotion.get("loneliness", 0.0)
    yearning = emotion.get("yearning", 0.0)
    guilt = emotion.get("guilt", 0.0)
    anger = emotion.get("anger", 0.0)
    numbness = emotion.get("numbness", 0.0)
    guidance_mode = strategy.get("guidance_mode", "none")

    if strategy.get("name") == "emotional_holding":
        return {
            "key": "name_the_grief",
            "title": "给此刻的难过留一句话",
            "category": "情绪承接",
            "prompt": "不用解释原因，只写下现在身体或心里最明显的一点感受。",
            "placeholder": "例如：我胸口很闷，今天只想安静一会儿。",
            "rationale": "当用户还不想展开回忆或建议时，先把当下感受命名出来，本身就是一种温和的安放。",
            "allow_save": True,
        }

    if strategy.get("name") == "behavioral_activation_support":
        return {
            "key": "one_basic_care_action",
            "title": "只选一个今天能做的小动作",
            "category": "行为激活",
            "prompt": "从喝水、吃一点东西、洗脸、出门站一分钟、给一个人发消息里，选一个最不费力的动作写下来。",
            "placeholder": "例如：我先喝半杯水，然后把窗帘拉开一点。",
            "rationale": "当低落和退缩影响日常功能时，先恢复一个很小的现实动作，比要求自己立刻好起来更可行。",
            "allow_save": True,
        }

    if guilt >= 0.55 or strategy.get("name") == "self_compassion":
        return {
            "key": "compassion_letter",
            "title": "把自责换成一封温柔的短笺",
            "category": "自我宽恕",
            "prompt": "写下你已经为它认真做过的一件事，哪怕很小。",
            "placeholder": "例如：我曾经每天陪它、给它准备喜欢的东西，也在努力照顾它。",
            "rationale": "愧疚感强时，先看见自己曾经付出的爱，可以帮助松动“全都是我的错”的想法。",
            "allow_save": True,
        }

    if anger >= 0.55 or strategy.get("name") == "anger_validation":
        return {
            "key": "anger_container",
            "title": "给不公平感一个安全出口",
            "category": "情绪承接",
            "prompt": "把此刻最想说的一句“不公平”写下来，不需要马上讲道理。",
            "placeholder": "例如：我真的觉得这太突然、太不公平了，我还没准备好失去它。",
            "rationale": "愤怒和不甘也是哀伤的一部分，先安全表达出来，比急着压下去更温和。",
            "allow_save": True,
        }

    if numbness >= 0.55 or strategy.get("name") == "numbness_grounding":
        return {
            "key": "grounding_now",
            "title": "回到此刻的三个小事实",
            "category": "稳定练习",
            "prompt": "写下你现在能看见、听见或触碰到的三个东西。",
            "placeholder": "例如：桌上的杯子、窗外的声音、手心的温度。",
            "rationale": "麻木和不真实感明显时，先回到身体和环境，有助于让情绪慢慢落地。",
            "allow_save": True,
        }

    if guidance_mode == "coping_guidance":
        return {
            "key": "one_small_step",
            "title": "今天只做一个很小的照顾动作",
            "category": "行为激活",
            "prompt": "写下今天你愿意为自己做的一件很小的事。",
            "placeholder": "例如：喝一杯水、吃一点东西、洗个热水澡、给一个人发消息。",
            "rationale": "当用户主动寻求办法时，低负担的小行动比宏大的建议更容易真正落地。",
            "allow_save": True,
        }

    if loneliness >= 0.55:
        return {
            "key": "support_person",
            "title": "给孤单留一个出口",
            "category": "现实支持",
            "prompt": "写下一个你可以稍微靠近的人，哪怕只是发一句很短的消息。",
            "placeholder": "例如：我可以问室友今晚能不能一起吃饭。",
            "rationale": "孤单感明显时，系统会鼓励用户把支持从虚拟陪伴延伸到现实关系。",
            "allow_save": True,
        }

    if sadness >= 0.55 and yearning < 0.5:
        return {
            "key": "self_compassion",
            "title": "给现在的自己一句宽一点的话",
            "category": "自我宽恕",
            "prompt": "如果是一个朋友正在经历同样的失去，你会怎么温柔地对他说？",
            "placeholder": "例如：你已经很认真地爱过它了，不需要把所有遗憾都变成责备。",
            "rationale": "强烈悲伤常常会伴随自责，换一个对朋友说话的角度有助于降低自我攻击。",
            "allow_save": True,
        }

    if grief_stage == "integration":
        return {
            "key": "meaning_bridge",
            "title": "把一段回忆带进以后的生活",
            "category": "意义重建",
            "prompt": "写下它留给你的一个习惯、改变或提醒。",
            "placeholder": "例如：它让我知道，安静陪着一个人也可以是很深的爱。",
            "rationale": "当用户开始重新连接生活时，意义重建能帮助回忆成为继续生活的一部分。",
            "allow_save": True,
        }

    return {
        "key": "one_sentence_to_pet",
        "title": "留下一句今天想对它说的话",
        "category": "持续性联结",
        "prompt": "如果只能给它留下一句话，你今天最想说什么？",
        "placeholder": "例如：今天又想起你在门口等我的样子，我还是很想你。",
        "rationale": "强烈思念不需要被立刻压下去，把一句话安放下来，本身就是一种温和的纪念。",
        "allow_save": True,
    }
