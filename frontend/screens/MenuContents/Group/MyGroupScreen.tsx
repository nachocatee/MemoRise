import { useEffect, useState, useRef } from "react";
import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import axios from "axios";
import {
  View,
  Text,
  Image,
  Pressable,
  ScrollView,
  Animated,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { useSelector } from "react-redux";

import SmallBtn from "../../../components/Button/SmallBtn";
import GroupBox from "../../../components/GroupBox";
import GoBackHeader from "../../../components/Header/GoBackHeader";
import { RootState } from "../../../store/store";
import { BACKEND_URL } from "../../../util/http";
import { styles } from "./GroupStyle";
import AlertModal from "../../../components/Modal/AlertModal";
import { BlurView } from "@react-native-community/blur";

type RootStackParamList = {
  FindGroup: undefined;
  GroupDetail: GroupDetailParams;
};

export type GroupData = {
  teamSeq: number;
  teamName: string;
  myProfile: string;
  ownerProfile: string;
  memberProfiles: string[];
  owner: boolean;
}[];

type GroupDetailParams = {
  teamSeq: number;
  userSeq: number;
};

const MyGroupScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();
  const [groupData, setGroupData] = useState<GroupData | null>(null);
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 그룹 나가기
  const [exitTeamSeq, setExitTeamSeq] = useState(0);

  const exitGroupHandler = () => {
    axios({
      method: "DELETE",
      url: BACKEND_URL + `/teams/${exitTeamSeq}/${userId}`,
    })
      .then((res) => {
        setGroupData((prevData) => {
          if (prevData) {
            return prevData.filter((group) => group.teamSeq !== exitTeamSeq);
          }
          return null;
        });
      })
      .catch((err) => {
        console.log(err);
      });
  };

  // 나가기 확인 모달 (그룹장)
  const [isHostExitModal, setHostExitModal] = useState(false);

  // 나가기 확인 모달 (참가자)
  const [isMemberExitModal, setMemberExitModal] = useState(false);

  const openExitModal = (teamSeq: number, isHost: boolean) => {
    if (isHost) {
      setHostExitModal(true);
    } else {
      setMemberExitModal(true);
    }
    setExitTeamSeq(teamSeq);
  };

  // 취소 버튼 눌렀을 때
  const closeExitModal = () => {
    setHostExitModal(false);
    setMemberExitModal(false);
  };

  // 확인 버튼 눌렀을 때
  const exitConfirm = () => {
    setHostExitModal(false);
    setMemberExitModal(false);
    exitGroupHandler();
  };

  // 그룹 상세 페이지로 이동하기
  const goDetailHandler = (teamSeq: number) => {
    if (!isEditGroup) {
      navigation.navigate("GroupDetail", {
        teamSeq: teamSeq,
        userSeq: userId,
      });
    }
  };

  // 내 그룹 목록 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          url: BACKEND_URL + `/user/${userId}/my-teams`,
        });
        setGroupData(res.data);
        console.log(res.data);
        console.log("조회 성공");
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, []);

  //  그룹 편집 버튼
  const [isEditGroup, setEditGroup] = useState(false);

  const EditGroupsHandler = () => {
    setEditGroup(!isEditGroup);
  };

  // 그룹 편집 애니메이션
  const shakeAnimation = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (isEditGroup) {
      startShake();
    } else {
      shakeAnimation.stopAnimation();
      shakeAnimation.setValue(0);
    }
  }, [isEditGroup]);

  const startShake = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(shakeAnimation, {
          toValue: 1,
          duration: 100,
          useNativeDriver: true,
        }),
        Animated.timing(shakeAnimation, {
          toValue: -1,
          duration: 100,
          useNativeDriver: true,
        }),
      ]),
      {
        iterations: -1, // 무한 반복
      }
    ).start();
  };

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>내 그룹</Text>
          {groupData && groupData[0] && (
            <>
              {isEditGroup ? (
                <SmallBtn onPress={EditGroupsHandler}>완료</SmallBtn>
              ) : (
                <SmallBtn onPress={EditGroupsHandler}>편집</SmallBtn>
              )}
            </>
          )}
        </View>
        {groupData && groupData[0] ? (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={styles.groupContainer}
          >
            {groupData.map((group, idx) => (
              <Animated.View
                key={idx}
                style={{
                  transform: [
                    {
                      rotate: shakeAnimation.interpolate({
                        inputRange: [-1, 1],
                        outputRange: ["-1deg", "1deg"], // 여기서 각도 조절
                      }),
                    },
                  ],
                }}
              >
                <GroupBox
                  teamName={group.teamName}
                  myProfile={group.myProfile}
                  memberProfiles={group.memberProfiles}
                  ownerProfile={group.ownerProfile}
                  owner={group.owner}
                  goDetailHandler={goDetailHandler}
                  teamSeq={group.teamSeq}
                />
                {isEditGroup && (
                  <Pressable
                    onPress={() => openExitModal(group.teamSeq, group.owner)}
                    style={styles.deleteContainer}
                  >
                    <Image
                      style={styles.delete}
                      source={require("../../../assets/image/delete.png")}
                    />
                  </Pressable>
                )}
              </Animated.View>
            ))}
          </ScrollView>
        ) : (
          <View style={styles.emptyContainer}>
            <Image
              style={styles.group}
              source={require("../../../assets/icons/group.png")}
            />
            <Text style={styles.empty}>참여한 그룹이 없습니다</Text>
            <Pressable
              style={({ pressed }) =>
                pressed
                  ? [styles.findGroupBtn, styles.pressed]
                  : styles.findGroupBtn
              }
              onPress={() => navigation.navigate("FindGroup")}
            >
              <Text style={styles.btnText}>그룹 찾기</Text>
            </Pressable>
          </View>
        )}
      </View>
      {/* (그룹장) 그룹 나가기 확인 모달 */}
      {isHostExitModal && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="light"
          blurAmount={4}
        >
          <AlertModal
            modalVisible={isHostExitModal}
            closeModal={closeExitModal}
            onConfirm={exitConfirm}
            contentText={`그룹장이 나가면\n그룹이 해체됩니다.\n정말 나가시겠습니까?`}
            btnText="나가기"
          />
        </BlurView>
      )}
      {/* (참가자) 그룹 나가기 확인 모달 */}
      {isMemberExitModal && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="light"
          blurAmount={4}
        >
          <AlertModal
            modalVisible={isMemberExitModal}
            closeModal={closeExitModal}
            onConfirm={exitConfirm}
            contentText="정말 나가시겠습니까?"
            btnText="나가기"
          />
        </BlurView>
      )}
    </LinearGradient>
  );
};

export default MyGroupScreen;
