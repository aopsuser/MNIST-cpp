'use strict';

const { GoogleGenerativeAI } = require('@google/generative-ai');

// Актуальные лёгкие/быстрые модели бесплатного тира Gemini, в порядке предпочтения.
// Если GEMINI_MODEL не задан явно, пробуем их по очереди (первая, что ответит без
// ошибки "модель не найдена", становится рабочей на весь прогон).
const DEFAULT_MODEL_CANDIDATES = [
  'gemini-2.0-flash',
  'gemini-1.5-flash',
  'gemini-1.5-flash-8b',
];

const SYSTEM_INSTRUCTION = `Ты — дружелюбный коллега-разработчик, который помогает поддерживать порядок в репозитории.
Твоя задача: посмотреть на заброшенную ветку или pull request и написать короткое,
человечное сообщение автору. Без канцелярита, без пассивно-агрессивного тона,
без фраз вроде "This PR has been inactive" или "Please respond within X days".
Пиши так, как будто ты реальный человек, который заметил старую ветку коллеги
и по-доброму спрашивает, что с ней делать. 2-4 предложения. Можно на русском или
английском — ориентируйся на язык коммит-мессаджей в контексте.`;

class AIClient {
  /**
   * @param {object} opts
   * @param {string} opts.apiKey
   * @param {string} [opts.model] - если не задано, пойдёт по DEFAULT_MODEL_CANDIDATES
   * @param {number} [opts.requestDelayMs=4500] - пауза между вызовами (защита от rate limit)
   * @param {number} [opts.maxRetries=5] - максимум ретраев при 429
   */
  constructor({ apiKey, model, requestDelayMs = 4500, maxRetries = 5 }) {
    if (!apiKey) {
      throw new Error(
        'GEMINI_API_KEY не задан. Получи бесплатный ключ на https://aistudio.google.com/app/apikey ' +
          'и добавь его в GitHub Secrets (для Action) или в .env (для локального теста).'
      );
    }

    this.genAI = new GoogleGenerativeAI(apiKey);
    this.requestedModel = model || null;
    this.resolvedModelName = model || null;
    this.requestDelayMs = requestDelayMs;
    this.maxRetries = maxRetries;
    this._lastCallAt = 0;
  }

  /**
   * Собирает промпт из данных о заброшенной ветке/PR.
   * @param {object} ctx
   * @param {string} ctx.branchName
   * @param {string[]} ctx.commitMessages
   * @param {string[]} ctx.changedFiles
   * @param {number} ctx.daysInactive
   * @param {boolean} ctx.hasConflicts
   * @returns {string}
   */
  buildPrompt({ branchName, commitMessages, changedFiles, daysInactive, hasConflicts }) {
    const commitsText =
      commitMessages && commitMessages.length
        ? commitMessages.map((m) => `- ${m}`).join('\n')
        : '(нет данных о коммитах)';

    const filesText =
      changedFiles && changedFiles.length
        ? changedFiles.slice(0, 20).join(', ') +
          (changedFiles.length > 20 ? `, и ещё ${changedFiles.length - 20} файлов` : '')
        : '(нет данных об изменённых файлах)';

    return `Вот информация о заброшенной ветке в репозитории:
Название ветки: ${branchName}
Последние коммиты:
${commitsText}
Изменённые файлы: ${filesText}
Дней без активности: ${daysInactive}
Есть конфликты с main: ${hasConflicts ? 'да' : 'нет'}

Предположи, какую задачу решала эта ветка, и напиши короткое дружелюбное сообщение
автору (2-4 предложения) с вопросом, актуальна ли ещё эта работа — стоит ли
продолжить, закрыть или смерджить. Пиши по-человечески, без канцелярита, без
пассивно-агрессивного тона, как будто это пишет коллега, а не бот. Не используй
фразы вроде "This PR has been inactive". Ответь только текстом сообщения, без
дополнительных пояснений и без кавычек вокруг текста.`;
  }

  /**
   * Генерирует персонализированное сообщение для одной ветки/PR.
   * Делает retry с exponential backoff при 429 (rate limit) и уважает паузу
   * между запросами, чтобы не упереться в лимиты бесплатного тира.
   * @param {object} ctx - см. buildPrompt
   * @returns {Promise<string>}
   */
  async generateMessage(ctx) {
    await this._respectRateLimit();

    const prompt = this.buildPrompt(ctx);
    const text = await this._callWithRetry(prompt);

    return this._sanitizeResponse(text, ctx);
  }

  /**
   * Ждёт нужное время с прошлого вызова, чтобы выдержать паузу requestDelayMs.
   */
  async _respectRateLimit() {
    const now = Date.now();
    const elapsed = now - this._lastCallAt;
    if (this._lastCallAt !== 0 && elapsed < this.requestDelayMs) {
      const waitMs = this.requestDelayMs - elapsed;
      await sleep(waitMs);
    }
    this._lastCallAt = Date.now();
  }

  /**
   * Разрешает имя модели: если пользователь явно задал GEMINI_MODEL — используем его.
   * Иначе пробуем кандидатов по очереди при первом реальном вызове.
   */
  async _resolveModel() {
    if (this.resolvedModelName) return this.resolvedModelName;

    const candidates = this.requestedModel
      ? [this.requestedModel]
      : DEFAULT_MODEL_CANDIDATES;

    let lastErr = null;
    for (const candidate of candidates) {
      try {
        const model = this.genAI.getGenerativeModel({
          model: candidate,
          systemInstruction: SYSTEM_INSTRUCTION,
        });
        // Пробный минимальный вызов, чтобы убедиться что модель существует и доступна
        await model.generateContent('ping');
        this.resolvedModelName = candidate;
        console.log(`[ai] Используем модель Gemini: ${candidate}`);
        return candidate;
      } catch (err) {
        lastErr = err;
        console.warn(`[ai] Модель "${candidate}" недоступна (${err.message}), пробуем следующую...`);
      }
    }

    throw new Error(
      `Не удалось найти рабочую модель Gemini среди [${candidates.join(', ')}]. ` +
        `Последняя ошибка: ${lastErr ? lastErr.message : 'неизвестна'}`
    );
  }

  /**
   * Вызывает Gemini API с retry и exponential backoff при 429/503.
   * @param {string} prompt
   * @returns {Promise<string>}
   */
  async _callWithRetry(prompt) {
    const modelName = await this._resolveModel();
    const model = this.genAI.getGenerativeModel({
      model: modelName,
      systemInstruction: SYSTEM_INSTRUCTION,
    });

    let attempt = 0;
    let lastErr = null;

    while (attempt <= this.maxRetries) {
      try {
        const result = await model.generateContent(prompt);
        const response = result.response;
        const text = response.text();
        if (!text || !text.trim()) {
          throw new Error('Gemini вернул пустой ответ');
        }
        return text.trim();
      } catch (err) {
        lastErr = err;
        const isRateLimit = isRateLimitError(err);
        const isRetryable = isRateLimit || isTransientError(err);

        if (!isRetryable || attempt === this.maxRetries) {
          break;
        }

        const backoffMs = computeBackoffMs(attempt, isRateLimit);
        console.warn(
          `[ai] Попытка ${attempt + 1}/${this.maxRetries} не удалась (${err.message}). ` +
            `Повтор через ${Math.round(backoffMs / 1000)}с...`
        );
        await sleep(backoffMs);
        attempt += 1;
      }
    }

    throw new Error(
      `Gemini API не ответил после ${attempt + 1} попыток. Последняя ошибка: ${
        lastErr ? lastErr.message : 'неизвестна'
      }`
    );
  }

  /**
   * Убирает возможные обёртки вроде кавычек или markdown-код-блоков,
   * которые модель иногда добавляет, несмотря на просьбу этого не делать.
   * Если после очистки текст пустой/подозрительно короткий — возвращает
   * безопасный fallback, а не бросает исключение (чтобы не ронять весь прогон
   * из-за одного странного ответа модели).
   */
  _sanitizeResponse(text, ctx) {
    let cleaned = text.trim();

    // Убираем обёртку в markdown-код-блок, если модель её добавила
    cleaned = cleaned.replace(/^```[a-z]*\n?/i, '').replace(/```$/i, '').trim();

    // Убираем обрамляющие кавычки
    if (
      (cleaned.startsWith('"') && cleaned.endsWith('"')) ||
      (cleaned.startsWith('«') && cleaned.endsWith('»'))
    ) {
      cleaned = cleaned.slice(1, -1).trim();
    }

    if (cleaned.length < 10) {
      console.warn(
        `[ai] Подозрительно короткий ответ модели для ветки "${ctx.branchName}", использую fallback-сообщение`
      );
      return this._fallbackMessage(ctx);
    }

    return cleaned;
  }

  /**
   * Простое запасное сообщение на случай, если модель вернула что-то
   * непригодное для использования — чтобы прогон не падал целиком.
   */
  _fallbackMessage(ctx) {
    return (
      `Привет! Заметил, что ветка \`${ctx.branchName}\` не обновлялась уже ${ctx.daysInactive} ` +
      `дней. Дай знать, актуальна ли ещё эта работа — можно продолжить, закрыть или смерджить.`
    );
  }
}

function isRateLimitError(err) {
  const status = err.status || err.statusCode || (err.response && err.response.status);
  if (status === 429) return true;
  const msg = (err.message || '').toLowerCase();
  return msg.includes('rate limit') || msg.includes('quota') || msg.includes('429');
}

function isTransientError(err) {
  const status = err.status || err.statusCode || (err.response && err.response.status);
  return status === 500 || status === 502 || status === 503 || status === 504;
}

function computeBackoffMs(attempt, isRateLimit) {
  // Rate limit ошибки бэкоффим агрессивнее, т.к. окно у бесплатного тира — минута
  const base = isRateLimit ? 15000 : 2000;
  const exponential = base * Math.pow(2, attempt);
  const jitter = Math.random() * 1000;
  return Math.min(exponential + jitter, 60000); // не больше 60с за раз
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

module.exports = { AIClient, DEFAULT_MODEL_CANDIDATES };
