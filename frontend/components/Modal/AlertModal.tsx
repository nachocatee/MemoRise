import React from "react";
import {
  View,
  Text,
  StyleSheet,
  GestureResponderEvent,
  Modal,
} from "react-native";

import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";
import ModalBtn from "../Button/ModalBtn";

type ModalProps = {
  modalVisible: boolean;
  closeModal: (event: GestureResponderEvent) => void;
  onConfirm: () => void;
  contentText: string;
  btnText: string;
};

const AlertModal: React.FC<ModalProps> = ({
  modalVisible,
  closeModal,
  onConfirm,
  contentText,
  btnText,
}) => {
  return (
    <Modal
      transparent={true}
      animationType="fade"
      visible={modalVisible}
      onRequestClose={closeModal}
    >
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <View style={styles.modalContainer}>
          <View style={styles.contentContainer}>
            <Text style={styles.title}>{contentText}</Text>
            <View style={styles.btnContainer}>
              <ModalBtn onPress={closeModal} color="white">
                취소
              </ModalBtn>
              <ModalBtn onPress={onConfirm} color={Colors.primary200}>
                {btnText}
              </ModalBtn>
            </View>
          </View>
        </View>
      </View>
    </Modal>
  );
};

export default AlertModal;

const styles = StyleSheet.create({
  modalContainer: {
    backgroundColor: "#E9E9E9",
    justifyContent: "center",
    alignItems: "center",
    width: calculateDynamicWidth(225) + 60,
    height: calculateDynamicWidth(113) + 40,
    borderRadius: calculateDynamicWidth(15),
    elevation: 4,
  },
  contentContainer: {
    width: calculateDynamicWidth(225),
    height: calculateDynamicWidth(113),
    justifyContent: "space-around",
  },
  btnContainer: {
    flexDirection: "row",
    justifyContent: "space-evenly",
  },
  title: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(18),
    color: Colors.text,
    marginBottom: calculateDynamicWidth(5),
  },
});
