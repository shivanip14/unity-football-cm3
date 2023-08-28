from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage
import uuid, json
from definitions import ROOT_DIR, ITERATION_SEPARATOR, SWEEP_SEPARATOR
import logging

logging.getLogger()
class CustomSideChannel(SideChannel):

    def __init__(self, stageNumber: str) -> None:
        super().__init__(uuid.UUID("421f0a70-4f87-11aa-a6bf-784f4387d1f7"))
        self.stageNumber = stageNumber
        with open(str(ROOT_DIR) + "/src/wincounts/" + stageNumber + "_win_counts_history.txt", "w") as f:
            f.write("")
            f.close()
        logging.info("CustomSideChannel initialized")

    def on_message_received(self, msg: IncomingMessage) -> None:
        msgString = msg.read_string()
        logging.info("Message received from Unity: " + msgString)
        jsonObject = json.loads(msgString)
        with open(str(ROOT_DIR) + "/src/wincounts/" + self.stageNumber + "_win_counts_history.txt", "a") as f:
            f.write(str(jsonObject['id']) + "," + str(jsonObject['blue']) + "," + str(jsonObject['purple']) + "\n") # each line corresponds to an episode
            f.close()

    def checkpoint_win_counts(self, stageNumber: str) -> None:
        with open(str(ROOT_DIR) + "/src/wincounts/" + stageNumber + "_win_counts_history.txt", "a") as f:
            f.write(ITERATION_SEPARATOR + "\n") # each section corresponds to an iteration
            f.close()

    def checkpoint_sweep(selfself, stageNumber: str) -> None:
        with open(str(ROOT_DIR) + "/src/wincounts/" + stageNumber + "_win_counts_history.txt", "a") as f:
            f.write(SWEEP_SEPARATOR + "\n")  # each section corresponds to a separate sweep run
            f.close()


