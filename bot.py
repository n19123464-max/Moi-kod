"""
Telegram-бот для корпоративной базы знаний ИТ-службы.

Стек: aiogram 3.x, режим polling, без БД и без ML.

Логика:
  - при старте сканируются все *.md файлы в папке скрипта (pathlib);
  - каждый файл разбирается на пункты регулярным выражением
    (форматы номеров: "1.", "1.2", а также markdown-заголовки "# 1.2");
  - для каждого пункта хранится: номер, заголовок, полный текст
    до следующего пункта, имя файла-источника;
  - поиск по свободному тексту: запрос очищается от стоп-слов,
    считается количество совпавших значимых слов + коэффициент
    схожести difflib.SequenceMatcher, пункт возвращается при
    combined_score > 1.15 и хотя бы одном общем слове;
  - ответ форматируется в HTML: экранирование html.escape,
    затем упрощённый markdown (**bold**, *italic*) конвертируется в HTML-теги.

Токен бота берётся из переменной окружения BOT_TOKEN (в коде его нет).
"""

import asyncio
import difflib
import html
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    ErrorEvent,
    FSInputFile,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kb_bot")

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
BOT_TOKEN = os.environ.get("BOT_TOKEN")

# Комбинированный порог score = matched_words + similarity_ratio
MATCH_SCORE_THRESHOLD = 1.15
MIN_COMMON_WORDS = 1
MIN_WORD_LEN = 3  # значимыми считаем слова длиннее этого числа символов

# ---------------------------------------------------------------------------
# Стоп-слова (русский язык)
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "ещё", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть",
    "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь",
    "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где",
    "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была",
    "сам", "чтоб", "без", "будто", "человек", "чего", "раз", "тоже",
    "себе", "под", "жизнь", "будет", "ж", "тогда", "кто", "этот", "того",
    "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один",
    "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда",
    "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об",
    "другой", "хоть", "после", "над", "больше", "тот", "через", "эти",
    "нас", "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между", "это", "эта", "эти", "этих", "этим",
    "также", "либо", "который", "которая", "которое", "которые",
    "которых", "которым", "которую", "мочь", "свой", "весь", "вся",
    "всё", "ему", "им", "ими", "нём", "ней", "неё", "него", "оно",
    "мои", "твой", "твоя", "твое", "твои", "наш", "наша", "наше", "наши",
    "ваш", "ваша", "ваше", "ваши", "какие", "какое", "каждый", "каждая",
    "каждое", "каждые", "любой", "любая", "любое", "любые", "иной",
    "другая", "другое", "другие", "пожалуйста", "здравствуйте", "привет",
    "спасибо", "подскажите", "скажите", "хочу", "нужно", "нужен",
    "нужна", "нужны", "имеет", "иметь", "являться",
    "является", "являются", "например", "данный", "данная", "данное",
    "данные", "именно", "т", "е", "др", "тд", "тп",
}

# ---------------------------------------------------------------------------
# Модель пункта базы знаний
# ---------------------------------------------------------------------------


@dataclass
class KBItem:
    number: str
    title: str
    text: str
    source_file: str


KB_ITEMS: List[KBItem] = []
MD_FILES: List[Path] = []

# Строка вида "1.", "1.2", "10.1.2", а также markdown-заголовок "# 1.2 ..."
ITEM_START_RE = re.compile(r"^#{0,6}\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")
WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def parse_md_file(path: Path) -> List[KBItem]:
    """Разбирает один .md файл на пункты по номерам."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Не удалось прочитать файл: %s", path)
        return []

    items: List[KBItem] = []
    current_number: Optional[str] = None
    current_title: str = ""
    current_lines: List[str] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = ITEM_START_RE.match(line)
        if match:
            if current_number is not None:
                items.append(
                    KBItem(
                        number=current_number,
                        title=current_title.strip(),
                        text="\n".join(current_lines).strip(),
                        source_file=path.name,
                    )
                )
            current_number = match.group(1)
            current_title = match.group(2).strip()
            current_lines = [current_title] if current_title else []
        else:
            if current_number is not None and line:
                current_lines.append(line)

    if current_number is not None:
        items.append(
            KBItem(
                number=current_number,
                title=current_title.strip(),
                text="\n".join(current_lines).strip(),
                source_file=path.name,
            )
        )

    return items


def load_knowledge_base() -> None:
    """Сканирует папку скрипта на .md файлы и наполняет KB_ITEMS."""
    global KB_ITEMS, MD_FILES

    MD_FILES = sorted(BASE_DIR.glob("*.md"))
    if not MD_FILES:
        logger.warning("В папке %s не найдено ни одного .md файла", BASE_DIR)

    all_items: List[KBItem] = []
    for md_path in MD_FILES:
        file_items = parse_md_file(md_path)
        logger.info("Файл %s: найдено пунктов — %d", md_path.name, len(file_items))
        all_items.extend(file_items)

    KB_ITEMS = all_items
    logger.info("Всего загружено пунктов базы знаний: %d", len(KB_ITEMS))


# ---------------------------------------------------------------------------
# Поиск по базе знаний
# ---------------------------------------------------------------------------


def extract_significant_words(text: str) -> set:
    words = {w.lower() for w in WORD_RE.findall(text)}
    return {w for w in words if len(w) > MIN_WORD_LEN and w not in STOP_WORDS}


def find_best_item(query: str) -> Optional[KBItem]:
    query_words = extract_significant_words(query)
    if not query_words or not KB_ITEMS:
        return None

    best_item: Optional[KBItem] = None
    best_score = 0.0
    query_lower = query.lower()

    for item in KB_ITEMS:
        item_full_text = f"{item.title} {item.text}"
        item_words = extract_significant_words(item_full_text)

        common_words = query_words & item_words
        matched_count = len(common_words)
        if matched_count < MIN_COMMON_WORDS:
            continue

        similarity = difflib.SequenceMatcher(
            None, query_lower, item_full_text.lower()
        ).ratio()

        combined_score = matched_count + similarity

        if combined_score > best_score:
            best_score = combined_score
            best_item = item

    if best_item is not None and best_score > MATCH_SCORE_THRESHOLD:
        return best_item
    return None


# ---------------------------------------------------------------------------
# Форматирование ответа в HTML
# ---------------------------------------------------------------------------


def markdown_to_html(escaped_text: str) -> str:
    """Конвертирует упрощённый markdown в HTML-теги.

    На вход подаётся уже экранированный через html.escape() текст,
    поэтому символы <, >, & в исходном контенте безопасны.
    """
    # Сначала жирный (двойные звёздочки), чтобы одинарные не съели их раньше времени
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped_text)
    # Затем курсив (одиночные звёздочки)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def format_item_answer(item: KBItem) -> str:
    safe_filename = html.escape(item.source_file)
    safe_number = html.escape(item.number)
    safe_text = markdown_to_html(html.escape(item.text))
    divider = "━" * 20

    return (
        f"📄 Документ: {safe_filename}\n"
        f"📎 Пункт {safe_number}\n"
        f"{divider}\n"
        f"{safe_text}"
    )


# ---------------------------------------------------------------------------
# Клавиатура и тексты
# ---------------------------------------------------------------------------

BTN_REGLAMENT = "📋 Прислать регламент"
BTN_HELP = "❓ Что умеет бот"
BTN_SEARCH = "🔍 Найти пункт"

main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_REGLAMENT)],
        [KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_SEARCH)],
    ],
    resize_keyboard=True,
)

HELP_TEXT = (
    "Я бот базы знаний ИТ-службы.\n\n"
    "Что я умею:\n"
    "• /start — приветствие и меню\n"
    "• /help — эта инструкция\n"
    "• /reglament — прислать все документы базы знаний (.md файлы)\n"
    "• Просто напишите вопрос своими словами — я найду подходящий пункт "
    "и укажу документ и номер пункта.\n\n"
    "Если подходящего пункта нет, я так и скажу, а не буду ничего придумывать."
)

SEARCH_HELP_TEXT = (
    "Напишите вопрос или ключевые слова обычным текстом, например:\n"
    "«как подать заявку в техподдержку» или «доступ к VPN».\n\n"
    "Я поищу совпадения в базе знаний и пришлю номер пункта и его текст."
)

# ---------------------------------------------------------------------------
# Хендлеры
# ---------------------------------------------------------------------------

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Здравствуйте! Я бот корпоративной базы знаний ИТ-службы.\n\n"
        "Выберите действие на клавиатуре ниже или сразу напишите вопрос — "
        "я поищу нужный пункт в документах.",
        reply_markup=main_kb,
    )


@router.message(Command("help"))
@router.message(F.text == BTN_HELP)
async def cmd_help(message: Message):
    await message.answer(HELP_TEXT, reply_markup=main_kb)


@router.message(F.text == BTN_SEARCH)
async def cmd_search_help(message: Message):
    await message.answer(SEARCH_HELP_TEXT, reply_markup=main_kb)


@router.message(Command("reglament"))
@router.message(F.text == BTN_REGLAMENT)
async def cmd_reglament(message: Message):
    if not MD_FILES:
        logger.error("Запрос регламента, но .md файлы не найдены в %s", BASE_DIR)
        await message.answer(
            "Документы базы знаний сейчас недоступны. Сообщите администратору.",
            reply_markup=main_kb,
        )
        return

    for md_path in MD_FILES:
        if not md_path.exists():
            logger.error("Файл пропал с диска: %s", md_path)
            continue
        try:
            await message.answer_document(
                FSInputFile(md_path),
                caption=md_path.name,
            )
        except Exception:
            logger.exception("Не удалось отправить файл: %s", md_path)
            await message.answer(
                f"Не получилось отправить файл {html.escape(md_path.name)}.",
            )


@router.message(F.text)
async def search_text(message: Message):
    query = message.text.strip()

    if not KB_ITEMS:
        await message.answer(
            "База знаний сейчас не загружена, поиск недоступен. "
            "Сообщите об этом администратору.",
            reply_markup=main_kb,
        )
        return

    item = find_best_item(query)
    if item is None:
        await message.answer(
            "К сожалению, не нашёл подходящего пункта по вашему запросу. "
            "Попробуйте переформулировать вопрос другими словами.",
            reply_markup=main_kb,
        )
        return

    await message.answer(format_item_answer(item), reply_markup=main_kb)


@router.errors()
async def errors_handler(event: ErrorEvent):
    logger.error(
        "Ошибка при обработке апдейта %s: %s",
        event.update,
        event.exception,
        exc_info=True,
    )
    return True


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


async def main():
    if not BOT_TOKEN:
        logger.error(
            "Не найдена переменная окружения BOT_TOKEN. Установите её перед запуском:\n"
            "  export BOT_TOKEN=ваш_токен_бота  (Linux/macOS)\n"
            "  set BOT_TOKEN=ваш_токен_бота      (Windows cmd)"
        )
        return

    load_knowledge_base()

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Бот запускается...")
    try:
        await dp.start_polling(bot)
    except Exception:
        logger.exception("Бот аварийно остановлен")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
