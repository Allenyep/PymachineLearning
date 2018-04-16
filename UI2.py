import wx
from shiyan import *

########################################################################
# set the file filter
wildcard1 = "All files (*.*)|*.*|" \
            "Python source (*.py; *.pyc)|*.py;*.pyc"
wildcard2 = "Python source (*.py; *.pyc)|*.py;*.pyc|" \
            "All files (*.*)|*.*"


########################################################################
class MyForm(wx.Frame):

    # -------------------------------------------------------------------
    # set the window layout
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          "Multi-file type wx.FileDialog Tutorial",
                          pos=(0, 0), size=(410, 335))
        # def the global variance
        global TxtCfn, Contents
        # layout the Frame
        panel = wx.Panel(self, wx.ID_ANY)
        TxtCfn = wx.TextCtrl(panel, pos=(15, 5), size=(200, 25))
        btnO = wx.Button(panel, label="Open", pos=(225, 5), size=(70, 25))
        btnS = wx.Button(panel, label="Run", pos=(300, 5), size=(70, 25))
        Contents = wx.TextCtrl(panel, pos=(15, 35), size=(360, 160),
                               style=wx.TE_MULTILINE | wx.HSCROLL)
        Results=wx.TextCtrl(panel,pos=(15,195),size=(360,100))
        # bind the button event
        btnO.Bind(wx.EVT_BUTTON, self.onOpenFile)
        btnS.Bind(wx.EVT_BUTTON, self.onRunFile)

    # -------------------------------------------------------------------
    # def [onOpenFile] function of the label [open]button
    def onOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            wildcard=wildcard1,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            tmp = ""
            # paths = dlg.GetPaths()
            paths = dlg.GetPaths()
            # print "You chose the following file(s):"
            for path in paths:
                tmp = tmp + path
            # set the value of TextCtrl[filename]
            TxtCfn.SetValue(tmp)
            print(tmp)
            # set the value to the TextCtrl[contents]
            file = open(TxtCfn.GetValue())
            Contents.SetValue(file.read())
            file.close()
        dlg.Destroy()

    # def onSaveFile function
    def onSaveFile(self, event):
        """
        Create and show the Save FileDialog
        """
        dlg = wx.FileDialog(self,
                            message="select the Save file style",
                            defaultFile="",
                            wildcard=wildcard2,
                            style=wx.FD_SAVE
                            )
        if dlg.ShowModal() == wx.ID_OK:
            filename = ""
            paths = dlg.GetPaths()
            # split the paths
            for path in paths:
                filename = filename + path
            # write the contents of the TextCtrl[Contents] into the file
            file = open(filename, 'w')
            file.write(Contents.GetValue())
            file.close()
            # show the save file path
            TxtCfn.SetValue(filename)
        dlg.Destroy()
    def onRunFile(self,event):
        tmp=TxtCfn.GetValue()
        print(tmp)
        run_model(tmp)
    # -----------------------------------------------------------------------


##########################################################################
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()