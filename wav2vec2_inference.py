import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor, pipeline, Wav2Vec2ForCTC

# Improvements: 
# - gpu / cpu flag
# - convert non 16 khz sample rates
# - inference time log

class Wave2Vec2Inference():
    def __init__(self,model_name,custom, hotwords=[], use_lm_if_possible = True):
        if use_lm_if_possible and custom == False:            
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            if not use_lm_if_possible and custom == False:
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            else:
                self.processor = Wav2Vec2Processor.from_pretrained("/home/nawaf/Projects/wav2vec2-live-main/wav2vec2-large-xlsr-WOLOF")
        if custom == False:
            self.model = AutoModelForCTC.from_pretrained(model_name)
        else:
            self.model  = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible
        self.custom = custom
    def buffer_to_text(self,audio_buffer):
        if(len(audio_buffer)==0):
            return ""

        
        if not self.custom:
            with torch.no_grad():
                inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)
                logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits            

            if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
                transcription = \
                    self.processor.decode(logits[0].cpu().numpy(),                                      
                                        hotwords=self.hotwords,
                                        #hotword_weight=self.hotword_weight,  
                                        output_word_offsets=True,                                      
                                    )                             
                confidence = transcription.lm_score / len(transcription.text.split(" "))
                transcription = transcription.text       
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                confidence = self.confidence_score(logits,predicted_ids)
        else:
            input_val = self.processor(audio_buffer, padding=True,sampling_rate=16_000).input_values

            input_dict = self.processor(input_val, return_tensors="pt", padding=True,sampling_rate=16_000)
            logits = self.model(input_dict.input_values.to("cpu")).logits
            
            pred_ids = torch.argmax(logits, dim=-1)[0]
            transcription = self.processor.decode(pred_ids)
            confidence =  0

        return transcription, confidence   

    def confidence_score(self,logits,predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self,filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2Inference("facebook/wav2vec2-large-960h-lv60-self")
    text = asr.file_to_text("test.wav")
    print(text)