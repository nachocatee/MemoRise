import React from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  GestureResponderEvent,
} from "react-native";

import Colors from "../../../constants/colors";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import ModalBtn from "../../Button/ModalBtn";

type ModalProps = {
  closeModal: (event: GestureResponderEvent) => void;
  onChangeText: (text: string) => void;
  value: string;
  onConfirm: () => void;
  invalid: boolean;
};

const PasswordInputModal: React.FC<ModalProps> = ({
  closeModal,
  onChangeText,
  value,
  onConfirm,
  invalid,
}) => {
  return (
    <View style={styles.modalContainer}>
      <View style={styles.contentContainer}>
        <Text style={styles.title}>비밀번호 입력</Text>
        <Text
          style={
            invalid ? [styles.content, styles.invalidText] : styles.content
          }
        >
          영문/숫자 4~8자리로 입력해 주세요.
        </Text>
        <TextInput
          style={invalid ? [styles.input, styles.invalidInput] : styles.input}
          onChangeText={onChangeText}
          value={value}
          maxLength={8}
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
    width: calculateDynamicWidth(240) + 60,
    height: calculateDynamicWidth(160) + 40,
    borderRadius: calculateDynamicWidth(15),
    elevation: 4,
  },
  contentContainer: {
    width: calculateDynamicWidth(240),
    height: calculateDynamicWidth(160),
    justifyContent: "center",
  },
  btnContainer: {
    flexDirection: "row",
    justifyContent: "space-evenly",
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
    fontSize: calculateDynamicWidth(13),
  },
  input: {
    borderBottomWidth: 2,
    borderBottomColor: Colors.text,
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(17),
    marginBottom: calculateDynamicWidth(30),
  },
  invalidInput: {
    borderBottomColor: Colors.red400,
  },
  invalidText: {
    color: Colors.red400,
  },
});
