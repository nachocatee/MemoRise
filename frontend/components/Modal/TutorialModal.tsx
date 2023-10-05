import React from "react";
import {
  View,
  Text,
  StyleSheet,
  GestureResponderEvent,
  Modal,
  Pressable,
  Image,
} from "react-native";

import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";
import ConfirmBtn from "../Button/ConfirmBtn";

type ModalProps = {
  modalVisible: boolean;
  closeModal: (event: GestureResponderEvent) => void;
  onlyCloseModal: (event: GestureResponderEvent) => void;
};

const TutorialModal: React.FC<ModalProps> = ({
  modalVisible,
  closeModal,
  onlyCloseModal,
}) => {
  return (
    <View style={styles.background}>
      <Modal
        transparent={true}
        animationType="slide"
        visible={modalVisible}
        onRequestClose={closeModal}
      >
        {/* 헤더 */}
        <View style={styles.header}>
          <Pressable>
            <Image
              source={require("../../assets/image/logo/logowhite.png")}
              style={styles.logo}
            />
          </Pressable>
          <Pressable style={styles.cancelContainer} onPress={onlyCloseModal}>
            <Image
              source={require("../../assets/icons/cancelwhite.png")}
              style={styles.cancel}
            />
          </Pressable>
        </View>

        <View style={styles.modalEmptyContainer}>
          <Text style={styles.helpTitle}>
            <Text style={{ color: Colors.primary200 }}>MemoRise</Text>는 두 가지
            단계를 거쳐{`\n`}메모를 등록할 수 있습니다!
          </Text>
          <View style={styles.helpContainer}>
            <Text style={styles.helpText}>1. 물체 등록하기</Text>
            <View style={{ marginLeft: calculateDynamicWidth(15) }}>
              <Text style={styles.helpContent}>
                a. 하단 중앙의 버튼{" "}
                <Image
                  source={require("../../assets/image/camerabtn.png")}
                  style={{
                    width: calculateDynamicWidth(20),
                    height: calculateDynamicWidth(20),
                  }}
                />{" "}
                을 누르세요
              </Text>
              <Text style={styles.helpContent}>
                b. 물체를 화면 중앙에 놓고 다각도로{`\n`} 비춰 주세요
              </Text>
            </View>
            <Text
              style={[
                styles.helpText,
                { marginTop: calculateDynamicWidth(15) },
              ]}
            >
              2. 메모 작성하기
            </Text>
            <View style={{ marginLeft: calculateDynamicWidth(15) }}>
              <Text style={styles.helpContent}>
                a. 등록한 물체에 메모를 추가해 주세요
              </Text>
              <Text style={styles.helpContent}>
                b. 메모에 사진을 첨부하고 친구를 태그할{`\n`} 수 있어요!
              </Text>
            </View>
          </View>
        </View>
        <View style={styles.btnContainer}>
          <ConfirmBtn onPress={closeModal}>시작하기</ConfirmBtn>
        </View>
      </Modal>
    </View>
  );
};

export default TutorialModal;

const styles = StyleSheet.create({
  background: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
  },
  header: {
    height: 97,
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
  },
  logo: {
    width: 189,
    height: 35,
  },
  cancelContainer: {
    position: "absolute",
    right: 25,
  },
  cancel: {
    width: 22.92,
    height: 22.92,
  },
  modalEmptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    marginTop: -calculateDynamicWidth(100),
  },
  helpTitle: {
    color: "white",
    fontFamily: "Pretendard-ExtraBold",
    fontSize: calculateDynamicWidth(24),
    textAlign: "center",
  },
  helpText: {
    color: "white",
    fontFamily: "Pretendard-SemiBold",
    fontSize: calculateDynamicWidth(18),
    marginVertical: calculateDynamicWidth(5),
  },
  helpContent: {
    color: "white",
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(18),
    marginVertical: calculateDynamicWidth(5),
  },
  helpContainer: {
    marginTop: calculateDynamicWidth(20),
  },
  btnContainer: {
    marginBottom: "10%",
    alignItems: "center",
  },
});
