import React from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  GestureResponderEvent,
} from "react-native";

import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";
import ModalBtn from "../Button/ModalBtn";

type ModalProps = {
  closeModal: (event: GestureResponderEvent) => void;
  onChangeText: (text: string) => void;
  value: string;
  onConfirm: () => void;
};

const PasswordInputModal: React.FC<ModalProps> = ({
  closeModal,
  onChangeText,
  value,
  onConfirm,
}) => {
  return (
    <View style={styles.modalContainer}>
      <View>
        <Text style={styles.title}>비밀번호 입력</Text>
        <Text style={styles.content}>영문/숫자 4~8자리로 입력해 주세요.</Text>
        <TextInput
          style={styles.input}
          onChangeText={onChangeText}
          value={value}
        />
        <View style={styles.btnContainer}>
          <ModalBtn onPress={closeModal} color="white">
            취소
          </ModalBtn>
          <ModalBtn onPress={onConfirm} color={Colors.primary200}>
            확인
          </ModalBtn>
        </View>
      </View>
    </View>
  );
};

export default PasswordInputModal;

const styles = StyleSheet.create({
  modalContainer: {
    backgroundColor: "#F5F5F5",
    justifyContent: "center",
    alignItems: "center",
    width: calculateDynamicWidth(273),
    height: calculateDynamicWidth(181),
    borderRadius: calculateDynamicWidth(15),
    elevation: 4,
  },
  btnContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
  },
  title: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(17),
    color: Colors.text,
    marginBottom: calculateDynamicWidth(5),
  },
  content: {
    fontFamily: "Pretendard-Light",
    color: "rgba(44, 44, 44, 0.5)",
    letterSpacing: -0.5,
  },
  input: {
    borderBottomWidth: 2,
    borderBottomColor: Colors.text,
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(17),
    marginBottom: calculateDynamicWidth(20),
  },
});
