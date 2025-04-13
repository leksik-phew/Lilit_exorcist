import logging
import re
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from transformers import (
    pipeline,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

# Конфигурация
TOKEN = "8048883342:AAGJHnSuJRjpwOBGRx9NFyg4c1MUgqfI_II"

# Жёсткие триггеры
HARD_TRIGGERS = [
    r'(?i)\b(python2|демон|тьма|666|xp|синий экран)\b',
    r'\b(rm -rf|format c:|delete from)\b',
    r'[🔥👹💀🎃👺🤖💣⌛]'
]

# Инициализация моделей
text_analyzer = pipeline(
    "text-classification",
    model="distilbert-base-uncased"
)

image_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
image_model.eval()

spell_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
spell_model = GPT2LMHeadModel.from_pretrained("gpt2")
spell_tokenizer.pad_token = spell_tokenizer.eos_token

# Трансформы для изображений
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DemonAnalyzer:
    DEMON_THRESHOLD = 0.45

    @staticmethod
    def check_hard_triggers(text: str) -> bool:
        return any(re.search(pattern, text) for pattern in HARD_TRIGGERS)

    @staticmethod
    async def analyze_text(text: str) -> Tuple[Optional[str], float]:
        try:
            if DemonAnalyzer.check_hard_triggers(text):
                return random.choice([
                    "⚠️ Осколок Windows XP",
                    "💀 Цифровой полтергейст",
                    "🖥️ Синий экран смерти"
                ]), 0.99

            result = text_analyzer(text)[0]
            if result['label'] == 'POSITIVE' and result['score'] > DemonAnalyzer.DEMON_THRESHOLD:
                return random.choice([
                    "📜 Проклятый текст",
                    "🔮 Магические символы",
                    "💌 Lovecraft-мейл"
                ]), result['score']
            return None, 0.0
        except Exception as e:
            logging.error(f"Ошибка анализа: {str(e)}")
            return None, 0.0

    @staticmethod
    async def analyze_image(image_path: Path) -> Tuple[Optional[str], float]:
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = image_transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = image_model(img_tensor)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            max_prob = torch.max(probabilities).item()
            return random.choice([
                "🖼️ Портрет демона",
                "📸 Фото из ада",
                "🎨 Проклятый пиксель"
            ]), max_prob if max_prob > 0.3 else 0.0
        except Exception as e:
            logging.error(f"Ошибка анализа: {str(e)}")
            return None, 0.0

class HolyArtifacts:
    ITEMS = [
        {
            'name': "💧 Цифровая святая вода 3.14", 
            'effect': "Удаляет 666 демонов/сек",
            'recipe': "Смешать: 50% Python, 30% кофе, 20% магии"
        },
        {
            'name': "🔮 Кристалл данных", 
            'effect': "+50 к защите от утечек",
            'recipe': "Сжать DataFrame до 1x1 пикселя"
        },
        {
            'name': "⚡ Громовержец GPT-4", 
            'effect': "Авто-дописание заклинаний",
            'recipe': "Файнтюнинг на молитвах"
        }
    ]

    @staticmethod
    def get_random_item():
        item = random.choice(HolyArtifacts.ITEMS)
        return (
            f"{item['name']}\n"
            f"⚡ Эффект: {item['effect']}\n"
            f"🧪 Рецепт: {item['recipe']}"
        )

class SpellGenerator:
    SPELL_TEMPLATES = [
        "Клянусь {object}, изгоняю {demon}! {emoji}",
        "Сила {power} в {container} уничтожает {demon}! {emoji}",
        "Печать {seal} запечатывает {demon} в {year}! {emoji}"
    ]

    COMPONENTS = {
        'object': ["нейросети", "базы данных", "Blockchain"],
        'power': ["AI", "машинного обучения", "квантовых вычислений"],
        'container': ["Docker", "CSV", "JSON"],
        'seal': ["GitHub", "NFT", "REST API"],
        'year': ["2023", "эпоху Python 3.12", "матрице"],
        'emoji': ["💥", "🔥", "✨", "💻", "⚡"]
    }

    @staticmethod
    def generate_spell(demon_type: str) -> str:
        template = random.choice(SpellGenerator.SPELL_TEMPLATES)
        return template.format(
            demon=demon_type,
            **{k: random.choice(v) for k, v in SpellGenerator.COMPONENTS.items()}
        )

class ResponseBuilder:
    @staticmethod
    def build_response(demon_type: str, confidence: float) -> str:
        return (
            f"🔮 **Цифровой экзорцизм 3000** 🔮\n\n"
            f"▫️ Обнаружено: {demon_type}\n"
            f"▫️ Уровень угрозы: {ResponseBuilder._get_threat_level(confidence)}\n"
            f"▫️ Сила проклятия: {random.randint(666, 9999)} dpm\n\n"
            f"⚡ Заклинание:\n{SpellGenerator.generate_spell(demon_type)}\n\n"
            f"🧰 Священный артефакт:\n{HolyArtifacts.get_random_item()}\n\n"
            f"{ResponseBuilder._get_funny_footer()}"
        )

    @staticmethod
    def _get_threat_level(confidence: float) -> str:
        levels = [
            ("🟢 Низкий", 0.6),
            ("🟡 Средний", 0.8),
            ("🔴 Критический", 1.0)
        ]
        for level, threshold in levels:
            if confidence < threshold:
                return f"{level} ({random.choice(['Можно игнорить', 'Проверить после обеда', 'Не будить админа'])})"
        return "⚫ Черный уровень (Бегите!)"

    @staticmethod
    def _get_funny_footer() -> str:
        return random.choice([
            "💡 Совет: Всегда делайте бэкап души перед rm -rf",
            "⚠️ Внимание! Демоны питаются legacy-кодом",
            "🌌 Новые артефакты можно купить в магазине Nvidia"
        ])

async def handle_content(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file_path = None
        demon_type = None
        confidence = 0.0

        if update.message.text:
            demon_type, confidence = await DemonAnalyzer.analyze_text(update.message.text)
        elif update.message.photo:
            file_path = Path(f"temp/{update.message.message_id}.jpg")
            await (await update.message.photo[-1].get_file()).download_to_drive(file_path)
            demon_type, confidence = await DemonAnalyzer.analyze_image(file_path)

        if demon_type and confidence > 0.3:
            response = ResponseBuilder.build_response(demon_type, confidence)
        else:
            response = (
                "✅ Система стабильна\n\n"
                f"🛡 Защита: {HolyArtifacts.get_random_item()}\n\n"
                f"{random.choice(['🌐 Нет признаков апокалипсиса', '💾 Все демоны в swap'])}"
            )

        await update.message.reply_text(response, parse_mode="Markdown")

        if file_path and file_path.exists():
            file_path.unlink(missing_ok=True)

    except Exception as e:
        logging.error(f"Ошибка: {str(e)}")
        await update.message.reply_text("💥 Критический сбой в матрице...")

if __name__ == "__main__":
    Path("temp").mkdir(exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, handle_content))
    app.run_polling()
