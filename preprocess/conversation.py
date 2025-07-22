# instruction = "The image are shown in sequence. Please generate analytical report on what will happen next in each region in Thailand (in term of geological event and disaster)? Please answer in Thai."
INSTRUCTION = """
  นี้คือบริบทที่เกียวข้องกับรูปภาพนี้:

  เอกสารจะถูกรายงานในวันที่ {} สำหรับทำเอกสาร {}

  ``` metadata ```
  {}

  ```รูปภาพถ่ายความกดอากาศ``` (มีทั้งหมด {} รูปภาพเรียงตามลำดับ)
  - หากพบ L ในรูปภาพจะหมายถึงความกดอากาศสูง ซึ่งมีโอกาสเกิดอากาศหนาว และมีโอกาสเกิดภัยแล้ง
  - หากพบ H ในรูปภาพจะหมายถึงความกดอากาศต่ำ ซึ่งมีโอกาสเกิดอากาศร้อน และมีโอกาสเกิดฝนตก

  ```รูปภาพถ่ายดาวเทียมแสดงความเคลื่อนไหวของกลุ่มเมฆ``` (มีทั้งหมด {} รูปภาพเรียงตามลำดับ)
  - โปรดตอบเป็นภาษาไทยเท่านั้น
  - หากพบ Cloudy ในรูปภาพจะหมายถึงเกิดกลุ่มเมฆขึ้น และหากพบ Typhoon หมายถึงเกิดพายุไต้ฝุ่นในช่วงเวลาดังกล่าว
  - เนื่องจากเป็นรายงานรายสัปดาห์ วันที่ที่ใช้รายงานจะต้องอยู่ในช่วง 7 วันก่อนที่เอกสารรายงาน (เช่น หากเอกสารรายงานในวันที่ 13 พฤษภาคม 2567 หมายความว่ารูปภาพจะอยู่ในช่วงวันที่ 06 พฤษภาคม 2567 ถึง 13 พฤษภาคม 2567 เท่านั้น)

  คำถาม: กรุณาทำรายงานวิเคราะห์เกี่ยวกับเหตุการณ์ทางธรณีวิทยาและภัยพิบัติที่จะเกิดขึ้นในแต่ละภูมิภาคของประเทศไทยและประเทศข้างเคียง
  คำตอบ:
"""

def convert_to_conversation(sample, instruction):
    conversation = [
        { "role": "user",
          "content":
          [
              item
              for i in range(len(sample["image"]))
              for item in (
                  {"type": "image", "image": sample["image"][i]},
              )
          ] + # images placeholder
          [{"type": "text", "text": instruction.format(sample['reportdate'], sample['filename'], sample['image_metadata'], len(sample['image']) - len(sample['image_metadata']), len(sample['image_metadata']))}] # instruction
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "messages" : conversation }

def convert_to_conversation_test(sample, instruction):
    conversation = [
        { "role": "user",
          "content":
          [{"type": "text", "text": instruction.format(sample['reportdate'], sample['filename'], sample['image_metadata'], len(sample['image']) - len(sample['image_metadata']), len(sample['image_metadata']))}] + # instruction
          [
              item
              for i in range(len(sample["image"]))
              for item in (
                  {"type": "image", "image": sample["image"][i]},
              )
          ] # images placeholder
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : ""} ]
        },
    ]
    return { "messages" : conversation }