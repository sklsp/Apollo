from ai_agent.tools import answer_questions, find_keywords, summarize


def main():
    try:
        with open('data/company_document.txt', encoding='utf-8') as f:
            doc = f.read()
    except FileNotFoundError:
        print('data/company_document.txt not found in project/data/')
        return

    try:
        print('\n--- SUMMARY (first 1000 chars) ---')
        print(summarize(doc)[:1000])

        print('\n--- KEYWORDS ---')
        print(find_keywords(doc))

        print('\n--- QA ---')
        print(answer_questions('What is the company mission?', doc))
    except Exception as e:
        print('Error during LLM call:', e)


if __name__ == '__main__':
    main()
