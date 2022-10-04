# _*_coding:utf-8_*_


import wx
import wx.media
import TextToSpeech
from TextToSpeech import TimePara
import cv2
# API
from FishRecognitionModel import FishRecogModel
from Dialogue_Flow import DialogueFlow


class UI_Frame(wx.Frame):
    def __init__(self):
        # layout
        super().__init__(parent=None, title="Fish Chat")
        # important attributes

        # Initiate UI
        self.InitUI()
        self.Center()

    def InitUI(self):
        panel = wx.Panel(self)
        # Headline font
        headline_font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        headline_font.SetPointSize(14)
        # content font
        content_font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        content_font.SetPointSize(9)
        self.sizer = wx.GridBagSizer(5, 5)

        # dialog notification
        boxsizer = wx.BoxSizer(wx.VERTICAL)
        headline = wx.StaticText(panel, label="Fish")
        headline.SetFont(headline_font)
        headline.SetForegroundColour('#4073FF')
        boxsizer.Add(headline, flag=wx.CENTER | wx.TOP, border=5)
        self.content = wx.StaticText(panel, label='Loading')
        self.content.SetFont(content_font)
        self.content.SetForegroundColour('#96C3EB')
        boxsizer.Add(self.content, flag=wx.CENTER | wx.TOP, border=5)

        self.sizer.Add(boxsizer, pos=(0, 0), flag=wx.TOP | wx.LEFT | wx.BOTTOM,
                       border=15)

        boxsizer_2 = wx.BoxSizer(wx.VERTICAL)
        headline_2 = wx.StaticText(panel, label="You")
        headline_2.SetFont(headline_font)
        headline_2.SetForegroundColour('#299438')
        boxsizer_2.Add(headline_2, flag=wx.CENTER | wx.TOP, border=5)
        self.content_2 = wx.StaticText(panel, label='Loading')
        self.content_2.SetFont(content_font)
        self.content_2.SetForegroundColour('#7ECC49')
        boxsizer_2.Add(self.content_2, flag=wx.CENTER | wx.TOP, border=5)

        self.sizer.Add(boxsizer_2, pos=(0, 1), flag=wx.TOP | wx.LEFT | wx.BOTTOM,
                       border=15)

        # Video place
        self.image_cover = wx.Image('COVER.jpg', wx.BITMAP_TYPE_ANY).Scale(480, 480)
        # Show this picture before clicking the bottom "Run Model"

        self.icon = wx.StaticBitmap(panel, bitmap=wx.Bitmap(self.image_cover))
        self.sizer.Add(self.icon, pos=(0, 4), flag=wx.TOP | wx.RIGHT | wx.ALIGN_RIGHT,
                       border=5)

        # border line
        line = wx.StaticLine(panel)
        self.sizer.Add(line, pos=(1, 0), span=(1, 5),
                       flag=wx.EXPAND | wx.BOTTOM, border=10)

        # buttons
        self.button1 = wx.Button(panel, label='Run Model')
        self.button1.Bind(wx.EVT_BUTTON, self.init_model)
        self.sizer.Add(self.button1, pos=(2, 0), flag=wx.LEFT, border=10)

        self.button2 = wx.Button(panel, label="Recode Voice")
        self.button2.Bind(wx.EVT_BUTTON, self.init_dialogue_flow)
        self.sizer.Add(self.button2, pos=(2, 1), flag=wx.LEFT)

        button3 = wx.Button(panel, label="Cancel")
        self.sizer.Add(button3, pos=(2, 4), span=(1, 1),
                       flag=wx.BOTTOM | wx.RIGHT | wx.ALIGN_RIGHT, border=10)

        self.sizer.AddGrowableCol(2)

        panel.SetSizer(self.sizer)
        self.sizer.Fit(self)

        # show the frame
        self.Show()

    def init_model(self, event):
        # using thread to avoid stuck
        import _thread
        _thread.start_new_thread(self._init_model, (event,))

    def _init_model(self, event=None):
        model = FishRecogModel("use")
        # cap = cv2.VideoCapture('./No_handnew.mp4')  # the video being tested
        cap = cv2.VideoCapture(0)

        hungry_time2 = 0
        tiring_time_duration = 0
        tiring_time2 = 0
        flag_tiring = 0
        flag_hungry = 0
        while (True):
            try:
                ret, frame = cap.read()
                img, hungry_time2, tiring_time2, tiring_time_duration, flag_hungry, flag_tiring \
                    = model.predict(frame, hungry_time2, tiring_time2, tiring_time_duration, flag_hungry, flag_tiring)
                height, width = img.shape[:2]
                image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pic = wx.Bitmap.FromBuffer(width, height, image1)
                self.icon.SetBitmap(pic)
                self.sizer.Fit(self)
            except:
                print('Please load the model first.')

    def init_dialogue_flow(self, event):
        # using thread to avoid stuck
        import _thread
        _thread.start_new_thread(self._init_dialogue_flow, (event,))

    def _init_dialogue_flow(self, event=None):
        df = DialogueFlow()
        df.record_voice()
        user_txt, dialogflow_txt = df.dialogflow_request()
        self.change_txt(user_txt, dialogflow_txt)
        df.play_audio()

    # change the notification context
    def change_txt(self, user_content, fish_content):
        self.content.SetLabel(fish_content)
        self.content_2.SetLabel(user_content)
        self.sizer.Fit(self)


if __name__ == "__main__":
    app = wx.App()
    frame = UI_Frame()
    app.MainLoop()
