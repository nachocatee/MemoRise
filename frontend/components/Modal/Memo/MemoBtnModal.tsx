import React from "react";
import {
  View,
  Text,
  Dimensions,
  StyleSheet,
  GestureResponderEvent,
  Pressable,
  Image,
} from "react-native";

import { calculateDynamicWidth } from "../../../constants/dynamicSize";

const screenWidth = Dimensions.get("window").width;
const screenHeight = Dimensions.get("window").height;

type ModalProps = {
  openMemoCreateModal: (event: GestureResponderEvent) => void;
  closeModal: (event: GestureResponderEvent) => void;
};

const MemoBtnModal: React.FC<ModalProps> = ({
  openMemoCreateModal,
  closeModal,
}) => {
  return (
    <>
      <View style={styles.memoBtnWrap}>
        <View style={styles.memoBtnContainer}>
          <Text style={styles.memoBtnText}>메모 조회하기</Text>
          <Pressable>
            <Image
              source={require("../../../assets/image/memoreadbtn.png")}
              style={styles.addBtn}
            />
          </Pressable>
        </View>
        <View style={styles.memoBtnContainer}>
          <Text style={styles.memoBtnText}>메모 작성하기</Text>
          <Pressable onPress={openMemoCreateModal}>
            <Image
              source={require("../../../assets/image/memocreatebtn.png")}
              style={styles.addBtn}
            />
          </Pressable>
        </View>
      </View>

      <Pressable onPress={closeModal} style={styles.btnContainer}>
        <Image
          source={require("../../../assets/image/cancelbtn.png")}
          style={styles.addBtn}
        />
      </Pressable>
    </>
  );
};

export default MemoBtnModal;

const styles = StyleSheet.create({
  btnContainer: {
    position: "absolute",
    bottom: calculateDynamicWidth(20),
    left: "50%",
    transform: [{ translateX: -calculateDynamicWidth(55) / 2 }],
  },
  addBtn: {
    width: calculateDynamicWidth(55),
    height: calculateDynamicWidth(55),
  },
  memoBtnContainer: {
    marginVertical: calculateDynamicWidth(5),
    alignItems: "center",
    justifyContent: "center",
  },
  memoBtnText: {
    color: "white",
    fontFamily: "Pretendard-SemiBold",
    fontSize: calculateDynamicWidth(18),
    marginRight: calculateDynamicWidth(10),
    position: "absolute",
    left: screenWidth / 2 - calculateDynamicWidth(150),
  },
  memoBtnWrap: {
    marginTop: screenHeight - calculateDynamicWidth(240),
  },
});
