import speech_recognition as sr
import os

# Pasta onde os arquivos de áudio estão localizados
audio_folder = "examples/"

# Listar arquivos de áudio na pasta
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

if not audio_files:
    print("Nenhum arquivo .m4a encontrado na pasta examples.")
else:
    print("Arquivos disponíveis na pasta 'examples/':")
    for i, file in enumerate(audio_files):
        print(f"{i + 1}. {file}")

    # Pergunta ao usuário qual arquivo deseja processar
    while True:
        try:
            choice = int(input("\nEscolha o número do arquivo que deseja processar: ")) - 1
            if 0 <= choice < len(audio_files):
                selected_audio = os.path.join(audio_folder, audio_files[choice])
                break
            else:
                print("Número inválido. Escolha um número da lista.")
        except ValueError:
            print("Por favor, insira um número válido.")

    # Processar o áudio selecionado
    def reconhecer_audio(arquivo_audio, idioma="en-US"):
        recognizer = sr.Recognizer()

        with sr.AudioFile(arquivo_audio) as source:
            print(f"\nProcessando o arquivo: {arquivo_audio}")
            audio = recognizer.record(source)

        try:
            texto = recognizer.recognize_google(audio, language=idioma)
            print("\n🔹 Texto reconhecido:", texto)
        except sr.UnknownValueError:
            print("\n❌ Não foi possível reconhecer a fala.")
        except sr.RequestError:
            print("\n❌ Erro ao conectar ao serviço de reconhecimento.")

    reconhecer_audio(selected_audio)
