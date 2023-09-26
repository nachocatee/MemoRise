import axios from "axios";
import React, { useEffect, useState } from "react";
import { View, Text, Image, Pressable, ScrollView } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { RouteProp } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { BACKEND_URL } from "../../../util/http";
import { RootStackParamList } from "../../../App";
import { styles } from "./GroupStyle";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import UserList from "../../../components/UserList";
import AlertModal from "../../../components/Modal/AlertModal";

type GroupDetailScreenProps = {
  route: RouteProp<RootStackParamList, "GroupDetail">;
  navigation: StackNavigationProp<RootStackParamList, "GroupDetail">;
};

type GroupData = {
  name: string;
  me: {
    userSeq: number;
    email: string;
    nickname: string;
    profile: string;
  };
  owner: {
    userSeq: number;
    email: string;
    nickname: string;
    profile: string;
  };
  members: [
    {
      userSeq: number;
      email: string;
      nickname: string;
      profile: string;
    }
  ];
  password: string;
};

const GroupDetailScreen: React.FC<GroupDetailScreenProps> = ({
  route,
  navigation,
}) => {
  const { teamSeq, userSeq } = route.params;
  const [groupData, setGroupData] = useState<GroupData | null>(null);

  // 그룹원 추방
  const kickUserHandler = (targetSeq: number) => {
    axios({
      method: "DELETE",
      url: BACKEND_URL + `/teams/${teamSeq}/kick`,
      data: {
        userSeq: userSeq,
        targetSeq: targetSeq,
      },
    })
      .then((res) => {
        console.log(res);
      })
      .catch((err) => {
        console.log(err);
      });
  };

  // 그룹원 추방 확인 모달
  const [isAlertModalVisible, setAlertModalVisible] = useState(false);
  const [targetUserId, setTargetUserId] = useState(0);
  const [targetUserName, setTargetUserName] = useState("");

  const openAlertModal = (targetSeq: number, targetName: string) => {
    setAlertModalVisible(true);
    setTargetUserId(targetSeq);
    setTargetUserName(targetName);
  };

  // 취소 버튼 눌렀을 때
  const closeAlertModal = () => {
    setAlertModalVisible(false);
  };

  // 확인 버튼 눌렀을 때
  const kickUserConfirm = () => {
    setAlertModalVisible(false);
    kickUserHandler(targetUserId);
  };

  // 그룹 상세 데이터 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          url: BACKEND_URL + `/teams/${teamSeq}/${userSeq}`,
        });
        setGroupData(res.data);
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, [groupData]);

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        {groupData && !groupData.owner ? (
          // 유저 == 그룹장일 때
          <>
            <View style={styles.titleContainer}>
              <Text style={styles.title}>{groupData.name}</Text>
              <Pressable
                onPress={() => navigation.navigate("GroupSetting")}
                style={styles.settingContainer}
              >
                <Image
                  style={styles.settingIcon}
                  source={require("../../../assets/icons/setting_sm.png")}
                />
                <Text style={styles.settingText}>그룹 설정</Text>
              </Pressable>
            </View>
            <View style={styles.memberContainer}>
              <Pressable
                onPress={() =>
                  navigation.navigate("InviteUser", {
                    teamSeq: teamSeq,
                    teamName: groupData.name,
                  })
                }
                style={[
                  styles.settingContainer,
                  { marginBottom: calculateDynamicWidth(25) },
                ]}
              >
                <View style={styles.memberImageContainer}>
                  <Image
                    style={styles.memberImagebg}
                    source={require("../../../assets/image/addmember.png")}
                  />
                </View>
                <Text style={styles.memberText}>초대하기</Text>
              </Pressable>
            </View>
            <ScrollView>
              {/* 내 정보 */}
              <View style={styles.userListContainer}>
                <UserList
                  profileUri={groupData.me.profile}
                  nickname={groupData.me.nickname}
                  email={groupData.me.email}
                />
                <Pressable>
                  <Image
                    style={styles.meIcon}
                    source={require("../../../assets/image/me.png")}
                  />
                </Pressable>
              </View>
              {/* 참가자 정보 */}
              {groupData.members.map((member, idx) => (
                <View key={idx} style={styles.userListContainer}>
                  <UserList
                    profileUri={member.profile}
                    nickname={member.nickname}
                    email={member.email}
                  />
                  <Pressable
                    onPress={() =>
                      openAlertModal(member.userSeq, member.nickname)
                    }
                  >
                    <Image
                      style={styles.cancelIcon}
                      source={require("../../../assets/icons/cancel_md.png")}
                    />
                  </Pressable>
                  {/* 그룹원 추방 확인 모달 */}
                  {isAlertModalVisible && (
                    <AlertModal
                      modalVisible={isAlertModalVisible}
                      closeModal={closeAlertModal}
                      onConfirm={kickUserConfirm}
                      contentText={`${targetUserName}님을 \n 추방하시겠습니까?`}
                      btnText="확인"
                    />
                  )}
                </View>
              ))}
            </ScrollView>
          </>
        ) : null}
      </View>
    </LinearGradient>
  );
};

export default GroupDetailScreen;
